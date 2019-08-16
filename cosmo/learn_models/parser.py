import os, sys
import argparse
from pathlib import Path
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-n", "--experiment-name", help="experiment name",\
                          type=str, required=True)
    required.add_argument("-a", "--arch", help="select model architecture",
                          action='store',
                          choices=['cosmoflow', 'resnext3d', 'resnext3dsmall'],
                          type=str, required=True)
    required.add_argument("-o", "--optim", help="select optimizer",\
                          action='store', choices=['sgd', 'adam'],\
                          type=str, required=True)
    required.add_argument("-i", "--input-data", \
                          help="directory containing input data", type=str, required=True)
    required.add_argument("-l", "--learning-rate", help="learning rate",\
                          type=float, required=True)
    required.add_argument("-b", "--batch-size", help="batch size", type=int, required=True)
    required.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)
    optional.add_argument("--no-pin-memory", help="Not using pin memory",
                          action='store_false', default=True, required=False)
    optional.add_argument("--gpu-device-id", help="GPU device id [=0]", type=int,
                          required=False, default=0)
    optional.add_argument("--num-workers", help="number of data loading workers [=4]",
                          type=int, required=False, default=4)
    optional.add_argument("--save", help="directory to save model checkpoints",
                          type=str, default="./save/", required=False)
    optional.add_argument("--output", help="directory to output model results",
                          type=str, default="./output/", required=False)
    optional.add_argument("--tensorboard", help="directory to save tensorboard \
                          summary", type=str, default="./runs/", required=False)
    optional.add_argument("--train-dev-split", help="(x, y): use x amount of data \
                          for training, and y amount for validation  training data.",\
                          type=float, default=None, nargs="+", required=False)
    optional.add_argument("--use-subset-data", help="filename, contains data \
                          indicies", type=str, default=None, required=False)
    optional.add_argument("--save-intervals", help="save the best model when \
                          the cumulative instances surpass the current save threshold.",\
                          type=int, default=None, required=False)
    optional.add_argument("--randseed", help="random seed for selecting a subset of the \
                          training data.", type=int, default=None, required=False)
    optional.add_argument('--local_rank', default=0, type=int)
    optional.add_argument('--skip_train', action='store_true', default=False)
    optional.add_argument('--backend', default='nccl', type=str,\
                          choices=['nccl', 'ddl'], required=False)
    args = parser.parse_args() 

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.backend, init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_device_id)

    return args
    
