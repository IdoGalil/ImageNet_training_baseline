from argparse import Namespace
import os

args = Namespace()

args.seed = 0
args.gpu_id = '0'
# For a different dataset than ImageNet, provide a different path below:
# args.data_dir = '/ImageNet'
args.data_dir = 'C:/Code/ImageNet'
args.num_workers = 4
args.dev = False
args.logger = True
