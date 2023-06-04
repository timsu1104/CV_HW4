import torch
import numpy as np
import os

from models import ModelWrapper
from datasets import build_dataset
from torch.utils.data import Subset, DataLoader
from utils import arg_parse

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import KFold
import wandb

def train(args, train_dataset, val_dataset, suffix: str):
    train_loader = DataLoader(
        train_dataset, 
        args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = ModelWrapper(
        args.num_classes, 
        args.epochs,
        args.lr_start, 
        args.lr_final, 
        args.wd, 
        args.alpha,
        args.gamma,
        args.backbone, 
        args.output_stride)
    
    train_tag = args.tag + suffix
    log_dir = f'./logs/{train_tag}'
    ckpt_dir = f'./ckpts/{train_tag}'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(name=train_tag, project="DeepLabv3+", save_dir=log_dir, log_model=True)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        num_nodes=1,
        strategy='ddp' if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1 else 'auto',
        default_root_dir=ckpt_dir,
        callbacks=[LearningRateMonitor(logging_interval='epoch', log_momentum=True)]
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    wandb.finish()

def main():
    args = arg_parse()
    torch.set_float32_matmul_precision('medium')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    dataset_aug, dataset_noaug, test_dataset = build_dataset(args)
    
    if args.test:
        train(args, dataset_aug, test_dataset, "_Test")
    else:
        for i, (train_idx, val_idx) in enumerate(k_fold.split(dataset_aug)):
            train_dataset = Subset(dataset_aug, train_idx)
            val_dataset = Subset(dataset_noaug, val_idx)
            train(args, train_dataset, val_dataset, f'_Fold{i}')

if __name__ == '__main__':
    main()