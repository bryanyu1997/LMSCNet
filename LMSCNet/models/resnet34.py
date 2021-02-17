import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torch.utils
import torchvision.models as models
import torchvision.transforms as tf
import pdb


class Resnet34(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    class_weights = self.get_class_weights().float()
    self.criti = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean')
    f = self.input_dimensions[1]

    encoder = models.resnet34(pretrained=True)
    self.preproc = nn.Sequential(
        nn.Conv2d(64, 64, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 7, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2),
    )
    self.layer1 = encoder.layer1
    self.layer2 = encoder.layer2
    self.layer3 = encoder.layer3
    self.layer4 = encoder.layer4
    self.decoder = Decoder()
    self.class_head = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, class_num*16, 1),
    )

  def forward(self, x):
    occ = x['3D_OCCLUDED']
    inputs = torch.cat((x['3D_OCCUPANCY'],x['3D_OCCLUDED'].unsqueeze(1)), dim=1)  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    inputs = inputs.permute(0, 1, 3, 2, 4)  # Reshaping to the right way for 2D convs [bs, H, W, D]
    B, Cinp, Z, Y, X =  inputs.shape
    inputs = inputs.reshape(B, Cinp*Z, Y, X)

    x0 = self.preproc(inputs)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    output = self.decoder(x1, x2, x3, x4)
    output = self.class_head(output).reshape(B, self.nbr_classes, Z//2, Y//2, X//2)
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

  def weights_init(self):
    pass

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''
    target = data['3D_LABEL']['1_1'].flatten().long()
    pred = scores['pred_semantic_1_1'].permute(0,2,3,4,1).flatten(end_dim=-2)
    mask = data['3D_OCCLUDED'].flatten()

    loss_free = F.cross_entropy(pred[mask==0], target[mask==0], ignore_index=255)
    loss_occ = self.criti(pred[mask==1], target[mask==1])
    loss_1_1 = loss_occ + 0.1 * loss_free

    loss = {
        'total': loss_1_1,
        'semantic_1_1': loss_1_1.detach(),
    }
    return loss

  def change_channel(self,layer,input_ch=1):
    ori = layer
    device = 'cuda'
    new = nn.Conv3d(in_channels=input_ch,out_channels=ori.out_channels,kernel_size=ori.kernel_size,stride=(1,2,2),padding=ori.padding,bias=ori.bias).to(device)  
    with torch.no_grad():
        new.weight[:,:] = ori.weight[:,:input_ch].cuda()

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

def Conv3d3x3BNReLU(inc, outc):
    return nn.Sequential(
        nn.Conv3d(inc, outc, 3, padding=1, bias=False),
        nn.BatchNorm3d(outc),
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
