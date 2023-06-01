import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .deeplab import DeepLab

import lightning.pytorch as pl
from torchmetrics import JaccardIndex

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * CE_loss
        return focal_loss.mean()

class ModelWrapper(pl.LightningModule):
    def __init__(
        self, 
        num_classes, 
        class_weight: torch.Tensor,
        epochs: int=1000,
        lr_start=1e-3,
        lr_final=1e-5,
        wd=1e-4,
        alpha=1,
        gamma=0,
        backbone='resnet101', 
        output_stride=16):
        
        super(ModelWrapper, self).__init__()
        self.model = DeepLab(num_classes, backbone=backbone, output_stride=output_stride)
        self.num_classes = num_classes
        self.class_weight = class_weight.cuda()
        self.criterion = FocalLoss(alpha, gamma, ignore_index=255)
        self.train_metric = JaccardIndex(task = 'multiclass', num_classes=num_classes, ignore_index=255, average='macro')
        self.val_metric = JaccardIndex(task = 'multiclass', num_classes=num_classes, ignore_index=255, average='macro')
        self.save_hyperparameters()
    
    # def get_iou(self, pred: torch.Tensor, gt: torch.Tensor):
    #     """
    #     pred: (B, H, W)
    #     gt: (B, H, W)
    #     """
    #     iou = []
    #     for label in range(self.num_classes):
    #         pred_mask = pred == label
    #         gt_mask = gt == label
    #         tp = torch.sum(pred_mask * gt_mask)
    #         fp = torch.sum(pred_mask) - tp
    #         tn = torch.sum(gt_mask) - tp
    #         denom = tp + fp + tn
    #         if denom > 0:
    #             iou[label] = tp / denom
    #     return torch.mean(iou)
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        out_img = self.model(image)
        loss = self.criterion(out_img, label)
        pred = out_img.max(1)[1]
        pixel_acc = torch.sum(pred == label) / label.numel()
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/pixel_acc", pixel_acc, sync_dist=True)
        self.train_metric.update(pred, label)
        return loss
    
    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train/mIoU', self.train_metric.compute(), sync_dist=True)
        self.train_metric.reset()
    
    def validation_step(self, batch, batch_idx):
        image, label = batch
        out_img = self.model(image)
        loss = self.criterion(out_img, label)
        pred = out_img.max(1)[1]
        pixel_acc = torch.sum(pred == label) / label.numel()
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/pixel_acc", pixel_acc, sync_dist=True)
        self.val_metric.update(pred, label)
        
    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val/mIoU', self.val_metric.compute(), sync_dist=True)
        self.val_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=[
                {'params': self.model.backbone.parameters(), 'lr': 0.1 * self.hparams.lr_start},
                {'params': self.model.ASPP.parameters(), 'lr': self.hparams.lr_start},
                {'params': self.model.decoder.parameters(), 'lr': self.hparams.lr_start},
            ], lr=self.hparams.lr_start, weight_decay=self.hparams.wd)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=(self.hparams.lr_final / self.hparams.lr_start) ** (1 / self.hparams.epochs))
        return [optimizer], [scheduler]