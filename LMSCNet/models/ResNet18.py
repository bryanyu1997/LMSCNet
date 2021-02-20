import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as tf

class ResNet18(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies, downsample=False):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    f = self.input_dimensions[1]
    self.downsample = downsample

    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

    self.encoder = models.resnet18(pretrained=True)
    if not self.downsample:
      self.stem = nn.Sequential(
          nn.Conv2d(64, 64, 3, padding=1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True)
      )
      self.class_head = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, class_num*32, 1),
    )
    else:
      self.stem = nn.Sequential(
          nn.Conv2d(64, 64, 3, padding=1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.AvgPool2d(2),  
      )
      self.class_head = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, class_num*16, 1),
    )
    self.decoder = Decoder()

  def forward(self, x):
    occ = x['3D_OCCLUDED']
    inputs = torch.cat((x['3D_OCCUPANCY'],x['3D_OCCLUDED'].unsqueeze(1)), dim=1)  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    inputs = inputs.permute(0, 1, 3, 2, 4)
    B, C, D, H, W = inputs.shape 
    x0 = inputs.reshape(B,-1,H,W)
    x0 = self.stem(x0)
    x1 = self.encoder.layer1(x0)
    x2 = self.encoder.layer2(x1)
    x3 = self.encoder.layer3(x2)
    x4 = self.encoder.layer4(x3)
    output = self.decoder(x1, x2, x3, x4)

    if not self.downsample:
      output = self.class_head(output)
      output = output.reshape(inputs.shape[0],self.nbr_classes,D,H,W)
    else:
      output = self.class_head(output)
      output = output.reshape(inputs.shape[0],self.nbr_classes,D//2,H//2,W//2)
      output = F.interpolate(output, scale_factor=2, mode='trilinear')

    # Take back to [W, H, D] axis order
    out_scale_1_1__3D = output.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    
    if not self.training:
      occ = occ.unsqueeze(dim = 1)
      freespace = (occ == 0).permute(0,2,3,4,1)    # BWHDC
      out_scale_1_1__3D = out_scale_1_1__3D.permute(0,2,3,4,1)  # BWHDC
      C = out_scale_1_1__3D.shape[-1]
      assignment = torch.zeros(1,C).to(out_scale_1_1__3D.device)
      assignment[:, 0] = 1

      out_scale_1_1__3D[freespace.squeeze(dim=4)] = assignment
      out_scale_1_1__3D = out_scale_1_1__3D.permute(0,4,1,2,3)

    scores = {'pred_semantic_1_1': out_scale_1_1__3D}

    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      try:
        nn.init.zeros_(m.bias)
      except AttributeError:
        return

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

    target = data['3D_LABEL']['1_1']
    scores = scores['pred_semantic_1_1'].permute(0,2,3,4,1)
    mask = data['3D_OCCLUDED']

    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)
    
    loss_free = criterion(scores[mask==0], target[mask==0].long())
    loss_occupied = criterion(scores[mask==1], target[mask==1].long())
    loss_1_1 = loss_free*0.1 + loss_occupied

    loss = {'total': loss_1_1, 'semantic_1_1': loss_1_1}

    return loss

  def change_channel(self,layer,input_ch=1):
    ori = layer
    device = 'cuda'
    new = nn.Conv3d(in_channels=input_ch,out_channels=ori.out_channels,kernel_size=ori.kernel_size,stride=(1,2,2),padding=ori.padding,bias=ori.bias).to(device)  
    with torch.no_grad():
        new.weight[:,:] = ori.weight[:,:1].cuda()
    return new

  def get_class_weights(self):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

    return weights

  def get_target(self, data):
    '''
    Return the target to use for evaluation of the model
    '''
    return {'1_1': data['3D_LABEL']['1_1']}
    # return data['3D_LABEL']['1_1'] #.permute(0, 2, 1, 3)

  def get_scales(self):
    '''
    Return scales needed to train the model
    '''
    scales = ['1_1']
    return scales

  def get_validation_loss_keys(self):
    return ['total', 'semantic_1_1']

  def get_train_loss_keys(self):
    return ['total', 'semantic_1_1']

def Conv2dBNReLU(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )

def conv2dbnrelu(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.lateral4 = conv2dbnrelu(512, 512)
        self.lateral3 = conv2dbnrelu(256, 256)
        self.lateral2 = conv2dbnrelu(128, 128)
        self.lateral1 = conv2dbnrelu(64, 64)
        self.fuse34 = conv2dbnrelu(512+256, 512)
        self.fuse23 = conv2dbnrelu(512+128, 512)
        self.fuse12 = conv2dbnrelu(512+64, 512)

    def forward(self, x1, x2, x3, x4):
        x1 = self.lateral1(x1)
        x2 = self.lateral2(x2)
        x3 = self.lateral3(x3)
        x4 = self.lateral4(x4)

        x = x4
        x = torch.cat([x3, F.interpolate(x, scale_factor=2, mode='bilinear')], 1)
        x = self.fuse34(x)
        x = torch.cat([x2, F.interpolate(x, scale_factor=2, mode='bilinear')], 1)
        x = self.fuse23(x)
        x = torch.cat([x1, F.interpolate(x, scale_factor=2, mode='bilinear')], 1)
        x = self.fuse12(x)

        return x