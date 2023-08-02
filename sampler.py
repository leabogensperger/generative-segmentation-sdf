import random
from types import SimpleNamespace
import imageio
import numpy as np
import argparse
import sys
import os
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import ImageDraw
import numpy.ma as ma

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import ImageFont, ImageDraw, Image
from torchvision.utils import make_grid
from torchmetrics import JaccardIndex, Dice, F1Score
import torchmetrics
from torchvision.utils import save_image
from torch.nn.functional import one_hot

from SimulationHelper.simulation import Simulation
from datasets.config_dl import config_dl
from models import ddpm

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--config", default="cfg/monuseg.yaml", type=str, help="path to .yaml config"
)
parser.add_argument("--seed", default=0, type=int, help="seed for reproducibility") # 1
args = parser.parse_args()

# Setting reproducibility
def set_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def store_gif(frames, frames_per_gif, load_path, sample_str=''):
    gif_name = load_path + "/samples/samples" + sample_str + ".gif"

    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])

def show_images(images, vmin=None, vmax=None, save_name="", overlay=None):
    """Shows the provided images as sub-pictures in a square"""
    alpha=0.6 if overlay is not None else 1. # alpha channel if additional overlay image is given
    if vmin is None:
        vmin = images.min().item() 
    if vmax is None:
        vmax = images.max().item() 

    if overlay is not None:
        overlay = overlay.detach().cpu().numpy()

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                if overlay is not None:
                    plt.imshow(overlay[idx][0], cmap="gray")
                    images[:,:,0,0] = vmax  # this is just for plotting!
                    images[:,:,0,1] = 1 
                    mask = np.ma.masked_where(images[idx][0] == 0, images[idx][0])  
                    plt.imshow(mask, alpha=alpha), plt.axis("off")
                else:
                    plt.imshow(images[idx][0], alpha=alpha, cmap="gray", vmin=vmin, vmax=vmax), plt.axis("off")
                idx += 1

    # Showing the figure
    plt.savefig(save_name, bbox_inches="tight", dpi=250)
    plt.close()


def compute_metrics(x,x_gt,thresh,corr_mode,num_classes):
    # compute IoU between thresholded x and x_gt
    x_thresh = torch.where(x > thresh, torch.zeros_like(x), torch.ones_like(x)).type(torch.int8).squeeze().cpu()
    x_gt_thresh = torch.where(x_gt > 0., torch.zeros_like(x_gt), torch.ones_like(x_gt)).type(torch.int8).squeeze().cpu()

    jaccard = JaccardIndex(task="binary")
    dice = Dice(task='binary',average='macro',num_classes=2,ignore_index=0) # dice=f1 in binary segmentation
    iou, dice = jaccard(x_thresh, x_gt_thresh), dice(x_thresh, x_gt_thresh)
    return iou, dice
   
        
def plot_all(x,cond,x_gt,img_cond,load_path,std_min,corr_mode,sample_str=''):
    sdf_min = x_gt.min().item() 
    sdf_max = x_gt.max().item() 

    show_images(x, vmin=sdf_min, vmax=sdf_max, save_name=load_path + "/samples/samples_" + str(sample_str) + ".png")
    if img_cond == 1:
        show_images(cond, save_name=load_path + "/samples/condition.png")
        show_images(x_gt, vmin=sdf_min, vmax=sdf_max, save_name=load_path + "/samples/groundtruth.png")

        x_thresh = torch.where(x > 3.*std_min, torch.zeros_like(x), torch.ones_like(x))
        show_images(x_thresh, save_name=load_path + "/samples/samples_thresholded_.png")

        x_gt_thresh = torch.where(x_gt > 0., torch.zeros_like(x_gt), torch.ones_like(x_gt))
        show_images(x_gt_thresh, save_name=load_path + "/samples/groundtruth_thresholded.png")

        # show thresholded maps on top of conditioning image
        vmax = x_gt.shape[1]
        show_images(x_thresh, save_name=load_path + "/samples/samples_thresholded_overlay.png", vmax=vmax, overlay=cond)
        show_images(x_gt_thresh, save_name=load_path + "/samples/groundtruth_thresholded_overlay.png", vmax=vmax, overlay=cond)

class Sampling:
    def __init__(self, scorenet, model_type, device, load_path, sz, noise_level_dict, img_cond, corr_mode, save_images=True):
        # general params
        self.scorenet = scorenet
        self.device = device
        self.load_path = load_path
        self.sz = sz
        self.img_cond = img_cond
        self.model_type = model_type
        self.corr_mode = corr_mode

        # if set to False, no images are saved
        self.save_images = save_images

        # noise level params
        self.s1, self.sL, self.L = noise_level_dict['s1'], noise_level_dict['sL'], noise_level_dict['L']
        self.sigmas = torch.tensor(np.exp(np.linspace(np.log(self.s1),np.log(self.sL), self.L))).type(torch.float32)

    def get_sigma(self,t):
        return self.sigmas[-1]*(self.sigmas[0]/self.sigmas[-1])**t

    def predictor_corrector(self, cond, x_gt, N, M, r, num_classes): 
        """
        Sample using reverse-time SDE (VE-SDE)
            N number of predictor steps
            M number of corrector steps 
            r "signal-to-noise" ratio
        """

        frames = []
        frames_thresh = []
        frames_all = []
        frames_per_gif = 100
        frame_idxs = np.linspace(0, N, frames_per_gif).astype(np.uint)
        
        t = torch.linspace(1-(1./N),0,N) # TODO: fix start at 1 or 1-dt
        sigma_t = self.get_sigma(t[0])
        x_list = []

        # initialize x and sample
        n_samples = x_gt.shape[0]
        x = self.sigmas[0]*torch.clip(torch.randn(n_samples,num_classes,self.sz,self.sz),-2.,2.).to(self.device) 

        with torch.no_grad():
            for i, t_curr in enumerate(t):
                if i % 20 == 0:
                    iou, dice = compute_metrics(x,x_gt,thresh=3*self.sigmas[-1].item(),corr_mode=self.corr_mode,num_classes=num_classes)
                    print('PC sampling it [%i]:\t IoU [%.6f], Dice [%.6f]' %(i,iou,dice))

                # set sigma(t) 
                sigma_t_prev = sigma_t.clone()
                sigma_t = self.get_sigma(t_curr)

                # get scores, sample noise
                if self.model_type == 'unet':
                    scores = self.scorenet(x,sigma_t_prev*torch.ones((n_samples,1)).to(self.device),img_cond=cond)
                elif self.model_type == 'tdv':
                    scores = self.scorenet.grad(torch.cat([x,cond],1),sigma_t_prev*torch.ones((n_samples,1,1,1)).to(self.device))[:,0:1]

                z = torch.clip(torch.randn_like(x),-2.,2.)
                tau = (sigma_t_prev**2 - sigma_t**2)

                # predictor step
                x = x + tau*scores
                x_list.append(x.detach().cpu())
                x += np.sqrt(tau)*z 

                # corrector steps
                for j in range(M):
                    # z = torch.randn_like(x)
                    z = torch.clip(torch.randn_like(x),-2.,2.)

                    # compute eps
                    if self.model_type == 'unet':
                        scores_corr = self.scorenet(x,sigma_t*torch.ones((n_samples,1)).to(self.device),img_cond=cond)
                    elif self.model_type == 'tdv':
                        scores_corr = self.scorenet.grad(torch.cat([x,cond],1),sigma_t*torch.ones((n_samples,1,1,1)).to(self.device))[:,0:1]

                    eps = 2*(r*torch.norm(z).item()/torch.norm(scores_corr).item())**2 
                    x = x + eps*scores_corr

                    x_list.append(x.detach().cpu())
                    x += np.sqrt(2*eps)*z

                if self.save_images and (i in frame_idxs or t_curr == 0): # TODO: if other samplers than PC are used, make sure that the gif is also generated for them
                    # Putting digits in range [0, 255]
                    normalized = x.clone()
                    if self.corr_mode == 'diffusion_ls':
                        normalized_thresh = torch.where(x > 3*self.sigmas[-1], torch.zeros_like(x), torch.ones_like(x))
                    elif self.corr_mode == 'diffusion':
                        normalized_thresh = torch.where(x < 0.5, torch.zeros_like(x), torch.ones_like(x))

                    for i in range(len(normalized)):
                        normalized[i] -= torch.min(normalized[i])
                        normalized[i] *= 255 / torch.max(normalized[i])

                        normalized_thresh[i] -= torch.min(normalized_thresh[i])
                        normalized_thresh[i] *= 255 / torch.max(normalized_thresh[i])

                    # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    frame = einops.rearrange(
                        normalized,
                        "(b1 b2) c h w -> (b1 h) (b2 w) c",
                        b1=int(n_samples**0.5),
                    )
                    frame = frame.cpu().numpy().astype(np.uint8)
                    frames.append(frame)

                    # append thresholded
                    frame_thresh = einops.rearrange(
                        normalized_thresh,
                        "(b1 b2) c h w -> (b1 h) (b2 w) c",
                        b1=int(n_samples**0.5),
                    )
                    frame_thresh = frame_thresh.cpu().numpy().astype(np.uint8)
                    frames_thresh.append(frame_thresh)

        # plotting
        if self.save_images:
            plot_all(x_list[-1],cond,x_gt,self.img_cond,load_path,std_min=cfg.SMLD.sigma_L,corr_mode=self.corr_mode,sample_str='pc')

        return x_list[-1], x_list

if __name__ == "__main__":
    with open(args.config) as file:
        yaml_cfg = yaml.safe_load(file)
        cfg = json.loads(json.dumps(yaml_cfg), object_hook=lambda d: SimpleNamespace(**d))

    device = torch.device("cuda")
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}"))

    set_seed(SEED=0)

    # set up dataloader, model
    train_dl, test_dl = config_dl(cfg)
    if cfg.model.type == 'unet':
        model = ddpm.Network(
            dim=cfg.model.dim,
            channels=cfg.model.n_cin,
            cond_channels=cfg.model.n_cin_cond,
            init_dim=cfg.model.n_fm,
            dim_mults=tuple(cfg.model.mults),
            embedding=cfg.model.embedding,
            img_cond=cfg.general.img_cond,
            with_class_label_emb=cfg.general.with_class_label_emb,
            class_label_cond=cfg.general.class_label_cond,
            num_classes=cfg.general.num_classes,
        ).to(device)

    else:
        raise ValueError('Unknown model type!')

    load_path = os.getcwd() + "/runs/" + cfg.general.modality

    if cfg.inference.latest:
        print("\nINFO: inference the lastest experiment!")
        load_path += "/" + sorted(os.listdir(load_path))[-1]
    else:
        print("\nINFO: inference from selected experiment, *not* the latest!")
        load_path += "/" + cfg.inference.load_exp

    print(f"Loading *latest* checkpoint from {load_path + '/models/'}")  
    if not os.path.exists(load_path + "/samples"):  # makedir for samples
        os.mkdir(load_path + "/samples")

    # load the model weights
    fnames = sorted(
        [fname for fname in os.listdir(load_path + "/models/") if fname.endswith(".pt")]
    )
    model.load_state_dict(torch.load(load_path + "/models/" + fnames[-1], map_location=device)["state_dict"],strict=True) # strict=False

    model.eval()
    print("\nModel loaded from %s" % (load_path + "/models/" + fnames[-1]))
    
    # sample and save generated images
    if cfg.general.corr_mode == "diffusion" or cfg.general.corr_mode == "diffusion_ls":

        noise_level_dict={'s1': cfg.SMLD.sigma_1, 'sL': cfg.SMLD.sigma_L, 'L': cfg.SMLD.n_steps}

        Sampler = Sampling(
            scorenet=model,
            model_type=cfg.model.type,
            device=device,
            load_path=load_path,
            sz=cfg.general.sz,
            noise_level_dict=noise_level_dict,
            img_cond=cfg.general.img_cond,
            corr_mode=cfg.general.corr_mode,
            save_images=True)

        # load conditioning image and ground truth
        it_test_dl = iter(test_dl) 
        batch = next(it_test_dl)
        cond = None if cfg.general.img_cond==0 else batch['mask'].to(device)
        x_gt = None if cfg.general.img_cond==0 else batch['image'].to(device)

        if cfg.SMLD.sampler == 'pc': # predictor corrector sampling
            print('\nPredictor-Corrector sampling with N=%i, M=%i, r=%f' %(cfg.SMLD.N,cfg.SMLD.M,cfg.SMLD.r))
            samples, samples_list = Sampler.predictor_corrector(cond=cond, x_gt=x_gt, N=cfg.SMLD.N,M=cfg.SMLD.M,r=cfg.SMLD.r, num_classes=cfg.model.n_cin)

        else:
            assert False, 'Specified sampler %s not implemented!' % cfg.SMLD.sampler

        # eval metrics
        iou, dice = compute_metrics(samples, x_gt.cpu(), corr_mode=cfg.general.corr_mode,thresh=3.*cfg.SMLD.sigma_L,num_classes=cfg.model.n_cin)
        print('\nFinal metrics: IoU [%f], Dice [%f]' %(iou,dice))
