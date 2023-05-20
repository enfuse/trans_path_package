from models.autoencoder import Autoencoder, PathLogger
from data.hmaps import GridData

import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

import argparse


def main(mode, run_name, proj_name):
    train_data = GridData(
        path='./TransPath_data/train',
        mode=mode
    )
    val_data = GridData(
        path='./TransPath_data/val',
        mode=mode
    )
    print('loading finished')
    train_dataloader = DataLoader(train_data, batch_size=64,
                        shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=64,
                        shuffle=False, num_workers=0, pin_memory=True)
    samples = next(iter(val_dataloader))
    
    model = Autoencoder(mode=mode)
    wandb_logger = WandbLogger(project=proj_name, name=run_name)
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=50,
        deterministic=False,
        callbacks=[PathLogger(samples, mode=mode)],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--proj_name', type=str, default='TransPath_runs')
    parser.add_argument('--seed', type=int, default=39)
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    main(
        mode=args.mode,
        run_name=args.run_name,
        proj_name=args.proj_name
    )
