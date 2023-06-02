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

def main():
    args = arg_parse()
    torch.set_float32_matmul_precision('medium')
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    dataset_aug, dataset_noaug, _ = build_dataset(args)
    class_weight = torch.from_numpy(np.load(args.class_weight)).float()
    
    for i, (train_idx, val_idx) in enumerate(k_fold.split(dataset_aug)):
        train_dataset = Subset(dataset_aug, train_idx)
        val_dataset = Subset(dataset_noaug, val_idx)
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
            class_weight, 
            args.epochs,
            args.lr_start, 
            args.lr_final, 
            args.wd, 
            args.alpha,
            args.gamma,
            args.backbone, 
            args.output_stride)
        
        os.makedirs(f'./logs/{args.tag}_Fold{i}', exist_ok=True)
        os.makedirs(f'./ckpts/{args.tag}_fold{i}', exist_ok=True)
        
        wandb_logger = WandbLogger(name=f"{args.tag}_Fold{i}", project="DeepLabv3+", save_dir=f'./logs/{args.tag}_Fold{i}', log_model=True)
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu',
            logger=wandb_logger,
            log_every_n_steps=10,
            check_val_every_n_epoch=5,
            num_nodes=1,
            strategy='ddp',
            default_root_dir=f'./ckpts/{args.tag}_fold{i}',
            callbacks=[LearningRateMonitor(logging_interval='epoch', log_momentum=True)]
        )
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )
        wandb.finish()

if __name__ == '__main__':
    main()