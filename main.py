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
import yaml

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from SimulationHelper.simulation import Simulation
from datasets.config_dl import config_dl
from models import ddpm
import trainer

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--config", default="cfg/monuseg.yaml", type=str, help="path to .yaml config" # glas, monuseg
)
args = parser.parse_args()

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

if __name__ == "__main__":
    # program arguments
    with open(args.config) as file:
        yaml_cfg = yaml.safe_load(file)
        cfg = json.loads(
            json.dumps(yaml_cfg), object_hook=lambda d: SimpleNamespace(**d)
        )

    device = torch.device("cuda")
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}"))

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

    # optimizer
    optim = Adam(model.parameters(), cfg.learning.lr)

    # Optionally, load a pre-trained model that will be further trained
    if cfg.general.resume_training:
        load_path = os.getcwd() + "/runs/" + cfg.general.modality 
        load_path += "/" + cfg.general.load_path + "/models/"
        fnames = sorted([fname for fname in os.listdir(load_path) if fname.endswith(".pt")])

        model.load_state_dict(
            torch.load(load_path + fnames[-1], map_location=device)["state_dict"],
            strict=False,
        )
        print("\nINFO: succesfully retrieved learned model params from specified cfg dir/epoch!")

        # load optimizer state dict
        optim.load_state_dict(torch.load(load_path + fnames[-1], map_location=device)["optimizer"])
        print("\nINFO: succesfully retrieved optim state dict specified cfg dir/epoch!")
        
    # network params
    print("\nNetwork has %i params" % count_parameters(model))

    # simulation
    sim_name = str(cfg.general.modality) 
    with Simulation(
        sim_name=sim_name, output_root=f'{os.path.join(os.getcwd(), "runs/")}'
    ) as simulation:
        writer = SummaryWriter(os.path.join(simulation.outdir, "tensorboard"))
        cfg.inference.load_exp = simulation.outdir.split("/")[-1]
        with open(os.path.join(simulation.outdir, "cfg.yaml"), "w") as f:
            yaml.dump({k: v.__dict__ for k, v in cfg.__dict__.items()}, f)

        # training
        if (cfg.general.corr_mode == "diffusion" or cfg.general.corr_mode == "diffusion_ls"):
            noise_level_dict={'s1': cfg.SMLD.sigma_1, 'sL': cfg.SMLD.sigma_L, 'L': cfg.SMLD.n_steps}

            trainer.TrainScoreNetwork(noise_level_dict,model_type=cfg.model.type,train_objective=cfg.SMLD.objective,loss_power=cfg.learning.loss,n_val=cfg.learning.n_val,val_dl=train_dl).do(
                model,
                train_dl,
                cfg.learning.epochs,
                cfg.learning.clip,
                optim=optim,
                device=device,
                simulation=simulation,
                writer=writer,
                img_cond=cfg.general.img_cond,
                class_label_cond=cfg.general.class_label_cond,
            )
