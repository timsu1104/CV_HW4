import argparse

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Cityscapes Trainer')

    # dataset
    parser.add_argument('--data-path', type=str, default='./data', help='Path to the dataset. ')
    parser.add_argument('--num-classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--class-weight', type=str, default='./data/class_weights.npy', help='Path to the class weight')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    # models
    parser.add_argument('--backbone', type=str, default='resnet101', help='Type of backbone')
    parser.add_argument('--output_stride', type=int, default=16, help='output_stride (8 or 16)')
    
    # training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    return args