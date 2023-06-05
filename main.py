import torch
import os
from tqdm.contrib import tzip
import cv2

from models import ModelWrapper
from datasets import build_dataset
from torch.utils.data import Subset, DataLoader
from utils import arg_parse

# Third Parties
from sklearn.model_selection import KFold
from cityscapesscripts.helpers.labels import labels
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

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
    
def eval(args, val_dataset):
    val_loader = DataLoader(
        val_dataset, 
        args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = ModelWrapper.load_from_checkpoint(args.eval_ckpt)
    
    output = pl.Trainer(
        accelerator='gpu', 
        strategy='ddp' if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1 else 'auto'
    ).test(model, dataloaders=val_loader)
    return output

def vis(args, val_dataset):
    
    vis_dataset = Subset(val_dataset, list(range(args.num_vis)))
    
    vis_loader = DataLoader(
        vis_dataset, 
        1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = ModelWrapper.load_from_checkpoint(args.eval_ckpt)
    trainer = pl.Trainer(
        accelerator='gpu', 
        strategy='ddp' if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1 else 'auto'
    )
    outputs = trainer.predict(model, dataloaders=vis_loader)
    
    categories = [l.name for l in labels if l.trainId != 255]
    colors = [l.color for l in labels if l.trainId != 255]
    MetadataCatalog.get("cityscapes").thing_classes = categories
    MetadataCatalog.get("cityscapes").thing_colors = colors
    MetadataCatalog.get("cityscapes").stuff_classes = categories
    MetadataCatalog.get("cityscapes").stuff_colors = colors
    
    for i, (output, img_path) in enumerate(tzip(outputs, val_dataset.file_names[:args.num_vis])):
        img = cv2.imread(img_path['img_path'], cv2.IMREAD_UNCHANGED)
        visualizer = Visualizer(img, MetadataCatalog.get("cityscapes"), instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_sem_seg(output[0].numpy(), area_threshold=1000, alpha=0.3)
        out = out.get_image()[:, :, ::-1]
        out_path = os.path.join(args.vis_path, f"vis_{i}.jpg")
        cv2.imwrite(out_path, out)
    
    return output

def main():
    args = arg_parse()
    torch.set_float32_matmul_precision('medium')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    dataset_aug, dataset_noaug, test_dataset = build_dataset(args)
    
    if args.test:
        train(args, dataset_aug, test_dataset, "_Test")
    elif args.eval:
        assert os.path.exists(args.eval_ckpt)
        eval(args, test_dataset)
    elif args.vis:
        assert os.path.exists(args.eval_ckpt)
        vis(args, test_dataset)
    else:
        for i, (train_idx, val_idx) in enumerate(k_fold.split(dataset_aug)):
            train_dataset = Subset(dataset_aug, train_idx)
            val_dataset = Subset(dataset_noaug, val_idx)
            train(args, train_dataset, val_dataset, f'_Fold{i}')

if __name__ == '__main__':
    main()