import argparse

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Cityscapes Trainer')
    
    parser.add_argument('--tag', type=str, default='DeepLabv3+', help='Tag of this run. ')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed (default: 42)')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use. Set for CUDA_VISIBLE_DEVICES')

    # dataset
    parser.add_argument('--data-path', type=str, default='./data', help='Path to the dataset. ')
    parser.add_argument('--num-classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    # models
    parser.add_argument('--backbone', type=str, default='resnet101', help='Type of backbone')
    parser.add_argument('--output_stride', type=int, default=16, help='output_stride (8 | 16)')
    
    # training
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs to be trained. ')
    parser.add_argument('--lr_start', type=float, default=3e-3, help='Learning rate at the beginning. ')
    parser.add_argument('--lr_final', type=float, default=3e-5, help='Learning rate in the end. ')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha in Focal Loss')
    parser.add_argument('--gamma', type=float, default=2, help='Beta in Focal Loss')
    
    # eval
    parser.add_argument('--test', action='store_true', help='Whether to use the original train-val split. ')
    parser.add_argument('--eval', action='store_true', help='Evaluation')
    parser.add_argument('--eval_ckpt', type=str, default='', help='The place of the checkpoint to be evaluated. ')
    
    # vis
    parser.add_argument('--vis', action='store_true', help='Visualization')
    parser.add_argument('--num_vis', type=int, default=1, help='Number of image to do Visualization')
    parser.add_argument('--vis_path', type=str, default='./visualizations', help='The place to save the visualization result. ')
    
    args = parser.parse_args()
    return args