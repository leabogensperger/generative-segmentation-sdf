import random
import imageio
import numpy as np
import argparse 
import sys
import os
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from SimulationHelper.simulation import Simulation
from datasets.config_dl import config_dl
from datasets.transform_factory import inv_normalize, transform_factory
from models import ddpm

class TrainScoreNetwork:
    def __init__(self, noise_level_dict, beta_dict, sde, model_type, train_objective, anneal_power=2, loss_power=2, n_val=8, val_dl=None):
        self.sde = sde 
        self.model_type = model_type

        if self.sde == 've':
            self.s1, self.sL, self.L = noise_level_dict['s1'], noise_level_dict['sL'], noise_level_dict['L']
            self.sigmas = torch.tensor(np.exp(np.linspace(np.log(self.s1),np.log(self.sL), self.L))).type(torch.float32)
            
            self.model_type = model_type
            self.anneal_power = anneal_power
            self.loss_power = loss_power
            self.train_objective = train_objective
            assert train_objective == 'disc' or train_objective == 'cont'

            if val_dl: # then use test dataloader
                val_batch = next(iter(val_dl))
                self.x_val = val_batch['image'][:n_val]
                self.cond_val = val_batch['mask'][:n_val]

                eta_val = torch.randn_like(self.x_val)
                self.used_sigmas_val = torch.linspace(self.sigmas[0],self.sigmas[-1], self.x_val.shape[0])[:,None,None,None]
                self.z_val = self.x_val + eta_val*self.used_sigmas_val

        elif self.sde == 'vp':
            self.beta1, self.betaT, self.T = beta_dict['beta1'], beta_dict['betaT'], beta_dict['T']
            self.betas = np.linspace(1.E-4, 0.02, 1000, dtype=np.float32)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.from_numpy(np.asarray([np.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]))

            if val_dl: # TODO: implement validation 
                val_batch = next(iter(val_dl))
                self.x_val = val_batch['image'][:n_val]
                pass 

        else:
            raise ValueError('Unknown SDE type!')

    def get_grad_norm(self, model):
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        norms = [p.grad.detach().abs().max().item() for p in parameters]
        return np.asarray(norms).max()

    def do(self, scorenet, dl, n_epochs, clip, optim, device, simulation, writer, img_cond, class_label_cond=False):
        if self.sde == 've':
            self._do_ve(scorenet, dl, n_epochs, clip, optim, device, simulation, writer, img_cond, class_label_cond)
        
        elif self.sde == 'vp':
            self._do_vp(scorenet, dl, n_epochs, clip, optim, device, simulation, writer, img_cond, class_label_cond)

    def _do_ve(self, scorenet, dl, n_epochs, clip, optim, device, simulation, writer, img_cond=0, class_label_cond=False):
        if img_cond == 0:
            self.cond_val = None 
        else:
            self.cond_val = self.cond_val.to(device)
        best_loss = float("inf")

        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            grad_norms_epoch = []

            for step, batch in enumerate(tqdm(dl, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                # Loading data
                x = batch['image'].to(device)
                cond = None if img_cond==0 else batch['mask'].to(device)
                lbl = None if class_label_cond is False else batch['label'].to(device).unsqueeze(1)
                n = len(x)

                # noise-conditional score network corruption 
                if self.train_objective == 'disc':
                    sigmas_idx = torch.randint(0, self.L, (n,))#.to(device)
                    used_sigmas = (self.sigmas[sigmas_idx][:,None,None,None]).to(device)
                elif self.train_objective == 'cont': # continuous training objective (SDE style)
                    t = torch.from_numpy(np.random.uniform(1e-5,1,(n,))).float() 
                    used_sigmas = (self.sigmas[-1]*(self.sigmas[0]/self.sigmas[-1])**t)[:,None,None,None].to(device)

                # noise corruption
                eta = torch.randn_like(x).to(device)
                z = x + eta*used_sigmas.to(device)

                # compute score matching loss
                target = 1/(used_sigmas**2) * (x-z)
                if self.model_type == 'unet':
                    scores = scorenet(z, used_sigmas.reshape(n,-1), img_cond=cond, class_lbl=lbl)
                elif self.model_type == 'tdv':
                    scores = scorenet.grad(torch.cat([z,cond],1), used_sigmas.reshape(n,1,1,1))[:,0:1]

                if step % 100 == 0: # Sanity Check. Whats going into the network?
                    with torch.no_grad():
                        scorenet.eval()
                        if self.x_val is not None: # always take same val/test batch
                            if self.model_type == 'unet':
                                scores_val = scorenet(self.z_val.to(device), self.used_sigmas_val.to(device).reshape(self.z_val.shape[0],-1), img_cond=self.cond_val, class_lbl=lbl)
                            elif self.model_type == 'tdv':
                                scores_val = scorenet.grad(torch.cat([self.z_val.to(device),self.cond_val],1), self.used_sigmas_val.to(device).reshape(self.z_val.shape[0],1,1,1))[:,0:1]

                            x_mmse_val = self.z_val.to(device) + self.used_sigmas_val.to(device)**2 * scores_val

                            # for multi-class plotting just take a random class
                            if self.x_val.shape[1] > 1:
                                class_idx = 4
                                x_val, z_val, x_mmse_val = self.x_val[:,class_idx][:,None], self.z_val[:,class_idx][:,None], x_mmse_val[:,class_idx][:,None]
                            else:
                                x_val, z_val = self.x_val, self.z_val 

                            all_stacked = torch.cat([
                                make_grid(x_val, nrow=x_val.shape[0], normalize=True, scale_each=True).cpu(),
                                make_grid(z_val, nrow=x_val.shape[0], normalize=True,scale_each=True).cpu(), 
                                make_grid(x_mmse_val, nrow=self.x_val.shape[0], normalize=True, scale_each=True).cpu()], dim=1)
                    
                        else: # check on random input data
                            x_mmse = z + used_sigmas**2*scores
                            all_stacked = torch.cat([
                                make_grid(x, nrow=x.shape[0], normalize=True, scale_each=True).cpu(),
                                make_grid(z, nrow=x.shape[0], normalize=True,scale_each=True).cpu(), 
                                make_grid(x_mmse, nrow=x.shape[0], normalize=True, scale_each=True).cpu()], dim=1)

                    # plot clean, noisy, and denoised (using Tweedie's formula)
                    if step % 100 == 0:
                        writer.add_image(f'training', all_stacked, global_step=epoch)
                        writer.flush()
                        scorenet.train()

                # Optimizing the MSE between the noise plugged and the predicted noise #
                loss_batches = ((torch.abs(target - scores))**self.loss_power).sum((-3,-2,-1))*used_sigmas.squeeze()**self.anneal_power # NOTE: L1 loss and anneal_power should match
                loss = loss_batches.mean()

                optim.zero_grad()
                loss.backward()
                
                if isinstance(clip,float):
                    torch.nn.utils.clip_grad_norm_(scorenet.parameters(), max_norm=clip, norm_type='inf')
                grad_norms_epoch.append(self.get_grad_norm(scorenet))

                optim.step()
                epoch_loss += loss.item() * len(x) / len(dl.dataset)
                

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.8f}"
            if epoch % 50 == 0:
                writer.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                
                writer.add_scalar(f'train/epoch_max_grad', np.asarray(grad_norms_epoch).max(), epoch)
                writer.add_scalar(f'train/epoch_mean_grad', np.asarray(grad_norms_epoch).mean(), epoch)
               
            # Storing the model
            if epoch % 5000 == 0: # save every 5000th epochs model, no matter what?
                checkpoint = {'state_dict': scorenet.state_dict()}
                simulation.save_pytorch(checkpoint, overwrite=False, subdir='models_sanity', epoch='_'+'{0:07}'.format(epoch))

            if best_loss > epoch_loss:
                best_loss = epoch_loss

                # save last 3 checkpoints
                if epoch > 0:
                    cp_dir = simulation._outdir + '/models'
                    if len([name for name in os.listdir(cp_dir) if os.path.isfile(os.path.join(cp_dir,name))]) == 3:
                        fnames = sorted([fname for fname in os.listdir(cp_dir) if fname.endswith('.pt')])
                        os.remove(os.path.join(cp_dir,fnames[0]))
                checkpoint = {'epoch': epoch, 
                            'state_dict': scorenet.state_dict(),
                            'optimizer': optim.state_dict()}
                simulation.save_pytorch(checkpoint, overwrite=False, epoch='_'+'{0:07}'.format(epoch))
                log_string += " --> Best model ever (stored)"

            print(log_string)

    def _do_vp(self, scorenet, dl, n_epochs, clip, optim, device, simulation, writer, img_cond, class_label_cond=False):
        best_loss = float("inf")
        mse = nn.MSELoss()

        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            grad_norms_epoch = []

            for step, batch in enumerate(tqdm(dl, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                # Loading data
                m = batch['image'].to(device)
                m *= 5. # TODO: comment if a clean data loader *without* scaling to [-0.2,0.2] is used - this is to get the mask to [-1,1] like the image
                x = None if img_cond==0 else batch['mask'].to(device)
                lbl = None if class_label_cond is False else batch['label'].to(device).unsqueeze(1)
                n = len(m)

                # noise corruption
                t = torch.randint(0, self.T, (n,)).to(device)
                a_bar = self.alpha_bars.to(device)[t]#.to(x.device)
                eta = torch.randn_like(x).to(device)
                m_noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * m + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

                # compute score matching loss
                if self.model_type == 'unet':
                    eta_estimated = scorenet(m_noisy, t.reshape(n,-1), img_cond=x, class_lbl=lbl)

                elif self.model_type == 'uvit':
                    eta_estimated = scorenet(m_noisy, t.reshape(n,-1), img_cond=x)

                elif self.model_type == 'tdv':
                    raise NotImplementedError

                # Optimizing the MSE between the noise plugged and the predicted noise #
                loss = mse(eta_estimated, eta)
                optim.zero_grad()
                loss.backward()

                if isinstance(clip,float):
                    torch.nn.utils.clip_grad_norm_(scorenet.parameters(), max_norm=clip, norm_type='inf')
                grad_norms_epoch.append(self.get_grad_norm(scorenet))

                optim.step()
                epoch_loss += loss.item() * len(x) / len(dl.dataset)
                
            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.8f}"
            if epoch % 10 == 0:
                writer.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                writer.add_scalar(f'train/epoch_max_grad', np.asarray(grad_norms_epoch).max(), epoch)
                writer.add_scalar(f'train/epoch_mean_grad', np.asarray(grad_norms_epoch).mean(), epoch)

            if best_loss > epoch_loss:
                best_loss = epoch_loss
                # save last 3 checkpoints
                if epoch > 0:
                    cp_dir = simulation._outdir + '/models'
                    if len([name for name in os.listdir(cp_dir) if os.path.isfile(os.path.join(cp_dir,name))]) == 3:
                        fnames = sorted([fname for fname in os.listdir(cp_dir) if fname.endswith('.pt')])
                        os.remove(os.path.join(cp_dir,fnames[0]))
                checkpoint = {'epoch': epoch, 
                            'state_dict': scorenet.state_dict(),
                            'optimizer': optim.state_dict()}
                simulation.save_pytorch(checkpoint, overwrite=False, epoch='_'+'{0:07}'.format(epoch))
                log_string += " --> Best model ever (stored)"
            print(log_string)