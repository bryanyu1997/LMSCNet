import os
import argparse
import torch
import torch.nn as nn
import sys
import time

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from LMSCNet.common.seed import seed_all
from LMSCNet.common.config import CFG
from LMSCNet.common.dataset import get_dataset
from LMSCNet.common.model import get_model


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='LMSCNet training')
  parser.add_argument(
    '--cfg',
    dest='config_file',
    default='',
    metavar='FILE',
    help='path to config file',
    type=str,
  )
  args = parser.parse_args()

  # https://github.com/pytorch/pytorch/issues/27588
  torch.backends.cudnn.enabled = False
  seed_all(0)

  # Read train configuration file
  train_f = args.config_file
  _cfg = CFG()
  _cfg.from_config_yaml(train_f)

  dataset = get_dataset(_cfg)
  model = get_model(_cfg, dataset['train'].dataset).cuda().eval()

  # measure
  num_parameter = sum([p.numel() for p in model.parameters()])
  print(f'Number of parameters: {num_parameter/1e6:.1f} M')

  run_second_lst = []
  x = {
    '3D_OCCLUDED': torch.randn(1, 256, 32, 256).cuda(),
    '3D_OCCUPANCY': torch.randn(1, 1, 256, 32, 256).cuda(),
  }
  with torch.no_grad():
    for i in range(50):
      s_time = time.time()
      _ = model(x)
      torch.cuda.synchronize()
      eps_time = time.time() - s_time
      run_second_lst.append(eps_time)
  run_time = sum(run_second_lst) / len(run_second_lst)
  print(f'FPS: {1/run_time:.2f}')

