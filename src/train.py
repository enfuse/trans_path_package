from models.autoencoder import Autoencoder, PathLogger, DemAutoencoder, DemPathLogger
from data.hmaps import GridData
from data.dems import DemData

import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import torch

import argparse
import multiprocessing

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(mode, run_name, proj_name, batch_size, max_epochs):
    train_data = GridData(
        path='./TransPath_data/train',
        mode=mode,
        img_size=256
    ) if mode != 'dem' else DemData(split='train')
    val_data = GridData(
        path='./TransPath_data/val',
        mode=mode,
        img_size=256
    ) if mode != 'dem' else DemData(split='val')
    resolution = (train_data.img_size, train_data.img_size)
    train_dataloader = DataLoader(  train_data, 
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=2,
                                    pin_memory=True)
    val_dataloader = DataLoader(    val_data, 
                                    batch_size=batch_size,
                                    shuffle=False, 
                                    num_workers=2,
                                    pin_memory=True)
    
    samples = next(iter(val_dataloader))
    
    model = Autoencoder(mode=mode, resolution=resolution) if mode != 'dem' else DemAutoencoder(resolution=resolution)
    callback = PathLogger(samples, mode=mode) if mode != 'dem' else DemPathLogger(samples)

    # Get the W&B API key from the environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    wandb_logger = WandbLogger(project=proj_name, name=f'{run_name}_{mode}', log_model='all')
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=max_epochs,
        deterministic=False,
        callbacks=[callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf', 'dem'], default='f')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--proj_name', type=str, default='TransPath_runs')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=160)
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') #fix for tesor blocks warning with new video card
    main(
        mode=args.mode,
        run_name=args.run_name,
        proj_name=args.proj_name,
        batch_size=args.batch,
        max_epochs=args.epoch,
    )
