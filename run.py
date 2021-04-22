import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3D, Classification, Unsup3D_Classifier


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--model', default="Unsup3D", type=str)
parser.add_argument('--rotated_angle', type=float)
parser.add_argument('--jitter_scale', type=float)
args = parser.parse_args()

## set up
model = eval(args.model)
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, model)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
