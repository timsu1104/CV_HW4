import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import lightning.pytorch as pl
from models import DeepLab
from datasets import build_dataset
from utils import arg_parse

class ModelWrapper(pl.LightningModule):
    def __init__(
        self, 
        num_classes, 
        class_weight: torch.Tensor,
        lr=1e-4,
        wd=1e-4,
        backbone='resnet101', 
        output_stride=16):
        
        super(ModelWrapper, self).__init__()
        self.model = DeepLab(num_classes, backbone=backbone, output_stride=output_stride)
        self.class_weight = class_weight.cuda()
        self.criterion = nn.CrossEntropyLoss(self.class_weight, ignore_index=255)
        self.lr = lr
        self.wd = wd
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        out_img = self.model(image)
        loss = self.criterion(out_img, label)
        pred = out_img.max(1)[1]
        pixel_acc = torch.sum(pred == label) / label.numel()
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_pixel_acc", pixel_acc, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, label = batch
        out_img = self.model(image)
        loss = self.criterion(out_img, label)
        pred = out_img.max(1)[1]
        pixel_acc = torch.sum(pred == label) / label.numel()
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_pixel_acc", pixel_acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

def main():
    args = arg_parse()
    torch.set_float32_matmul_precision('medium')
    
    train_loader, val_loader = build_dataset(args)
    
    class_weight = torch.from_numpy(np.load(args.class_weight)).float()
    model = ModelWrapper(args.num_classes, class_weight, args.lr, args.wd, args.backbone, args.output_stride)
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        log_every_n_steps=10,
        num_nodes=1,
        strategy='ddp'
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

if __name__ == '__main__':
    main()