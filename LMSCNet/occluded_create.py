import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import pdb

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from LMSCNet.common.seed import seed_all
from LMSCNet.common.config import CFG
from LMSCNet.common.dataset import get_dataset
from LMSCNet.common.model import get_model
from LMSCNet.common.logger import get_logger
from LMSCNet.common.optimizer import build_optimizer, build_scheduler
from LMSCNet.common.io_tools import dict_to
from LMSCNet.common.data_metric import Metrics
import LMSCNet.common.checkpoint as checkpoint

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

def parse_args():
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
  return args

def cart2polar(input_xyz, center):

    input_xyz = input_xyz - center
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.stack((rho, phi, input_xyz[..., 2]), axis=-1)

def train(dataset, _cfg):
  """
  Train a model using the PyTorch Module API.
  Inputs:
  - model: A PyTorch Module giving the model to train.
  - optimizer: An Optimizer object we will use to train the model
  - scheduler: Scheduler for learning rate decay if used
  - dataset: The dataset to load files
  - _cfg: The configuration dictionary read from config file
  - start_epoch: The epoch at which start the training (checkpoint)
  - logger: The logger to save info
  - tbwriter: The tensorboard writer to save plots
  Returns: Nothing, but prints model accuracies during training.
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  dtype = torch.float32  # Tensor type to be used

  dset = dataset['train']

  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
  nbr_iterations = len(dset)  # number of iterations depends on batchs size

  metrics = Metrics(dset.dataset.nbr_classes, nbr_iterations, ['1_1'])
  metrics.reset_evaluator()

  #max_dis = torch.sqrt(torch.sum((center[:2]-0)**2)).item()

  for t, (data, indices) in enumerate(dset):

    print(t)
    data = dict_to(data, 'cpu', dtype)
    target = data['3D_LABEL']['1_1'].squeeze().permute(0,2,1)
    #target[(target>0)&(target!=255)] = 1
    gt = data['3D_OCCLUDED'].squeeze().permute(0,2,1)
    inputs = data['3D_OCCUPANCY'].squeeze().permute(0,2,1)
    occluded = inputs.clone()
    center = torch.Tensor([0,127,0]).cuda()

    # Polar Experiment
    '''Xdim, Ydim, Zdim = inputs.shape 
    inputs_idx = inputs.nonzero()
    occluded_idx = torch.stack(torch.meshgrid(torch.arange(Xdim), torch.arange(Ydim), torch.arange(Zdim)), -1)
    inputs_dis = torch.sqrt(torch.sum((inputs_idx-center)**2,dim=1))
    occluded_dis = torch.sqrt(torch.sum((occluded_idx-center)**2,dim=1))
    
    polar_inputs = cart2polar(inputs_idx, center)
    polar_free = cart2polar(occluded_idx, center)

    for ang in polar_inputs[...,1]:
      for idx in (ang==polar_free[...,1]).nonzero():'''

    # Choose by near voxel
    '''center = torch.Tensor([0,127,0]).cpu()
    Xdim, Ydim, Zdim = inputs.shape 
    inputs_idx = inputs.nonzero()
    #occluded_idx = (inputs-1).nonzero()
    occluded_idx = torch.stack(torch.meshgrid(torch.arange(Xdim), torch.arange(Ydim), torch.arange(Zdim)), -1)
    occluded_idx = occluded_idx.reshape(Xdim*Ydim*Zdim, 3).float()
    inputs_dis = torch.sqrt(torch.sum((inputs_idx-center)**2,dim=1))
    occluded_dis = torch.sqrt(torch.sum((occluded_idx-center)**2,dim=1))

    for i, l_vox in enumerate(tqdm(occluded_idx)):
        match_vox = torch.sqrt(torch.sum((inputs_idx-l_vox.float())**2,dim=-1)).argmin(-1)
        if (inputs_dis[match_vox]) <= occluded_dis[i]:
            occluded[l_vox[0].long(), l_vox[1].long(), l_vox[2].long()] = 1'''

    Xdim, Ydim, Zdim = inputs.shape
    inputs_idx = inputs.nonzero().float().cuda()
    inputs_2_cam_dist = (inputs_idx - center).pow(2).sum(-1).sqrt()

    coor = torch.stack(torch.meshgrid(torch.arange(Xdim), torch.arange(Ydim), torch.arange(Zdim)), -1)
    coor = coor.reshape(Xdim*Ydim*Zdim, 3).float().cuda()
    coor_2_cam_dist = (coor - center).pow(2).sum(-1).sqrt()

    coor_2_closest_idx = (coor[:,None] - inputs_idx[None]).pow(2).sum(-1).sqrt().argmin(-1)
    closest_2_cam_dist = inputs_2_cam_dist[coor_2_closest_idx]
    freespace = (coor_2_cam_dist <= closest_2_cam_dist).float().reshape(Xdim, Ydim, Zdim)

    np.savez(os.path.join('../prediction/create_occluded','occluded'),occluded.data.cpu().numpy())
    np.savez(os.path.join('../prediction/create_occluded','occupancy'),inputs.data.cpu().numpy())
    np.savez(os.path.join('../prediction/create_occluded','target'),target.data.cpu().numpy())
    np.savez(os.path.join('../prediction/create_occluded','gt'),gt.data.cpu().numpy())
    import pdb
    pdb.set_trace() 

  return

def main():

  # https://github.com/pytorch/pytorch/issues/27588
  torch.backends.cudnn.enabled = False

  seed_all(0)

  args = parse_args()

  train_f = args.config_file

  # Read train configuration file
  _cfg = CFG()
  _cfg.from_config_yaml(train_f)

  # Replace dataset path in config file by the one passed by argument
  dataset_f = _cfg._dict['DATASET']['ROOT_DIR']

  dataset = get_dataset(_cfg)

  '''with open('/media/NFS/bryan/LMSCNet/SSC_out/statist/fp_dict.pickle', 'rb') as handle:
    fp = pickle.load(handle)
  with open('/media/NFS/bryan/LMSCNet/SSC_out/statist/tp_dict.pickle', 'rb') as handle:
    tp = pickle.load(handle)'''

  best_record = train(dataset, _cfg)

  exit()

if __name__ == '__main__':
  main()