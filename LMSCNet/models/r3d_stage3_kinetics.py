import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as tf

from .backbone_kinetics import VideoResNet
import pdb


class FastStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self, inp_channel, out_channel):
        super(FastStem, self).__init__(
            nn.Conv3d(inp_channel, out_channel, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True))


class r3d_stage3_kinetics(nn.Module):

  def __init__(self, class_num, input_dimensions, ch_lst=[16,32,128], stage_layer=[1,1,3]):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    #self.class_frequencies = class_frequencies

    self.channel_subtract = Conv3d3x3BN(3,1)
    self.encoder = VideoResNet(num_channels=ch_lst, stem=FastStem, layers=stage_layer, num_classes=400)
    #self.decoder = Decoder(ch_lst, 32)

  def forward(self, x):

    inputs = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    #inputs = torch.squeeze(inputs, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]
    #inputs = inputs.unsqueeze(1)
    inputs = inputs.permute(0,4,1,3,2)
    inputs = self.channel_subtract(inputs)

    output = self.encoder(inputs)
    
    '''feature_lst = []
    feature_lst.append(inputs.shape)
    x = self.encoder.stem(inputs)
    x = self.encoder.layer1(x)
    feature_lst.append(x)
    x = self.encoder.layer2(x)
    feature_lst.append(x)
    x = self.encoder.layer3(x)
    feature_lst.append(x)
    output = self.decoder(feature_lst)'''

    # Take back to [W, H, D] axis order
    #out_scale_1_1__3D = output.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    out_scale_1_1__3D = output

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

    target = data['3D_LABEL']
    device, dtype = target.device, target.dtype
    #class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    # Weighted-Tuning
    '''class_weights[2:9] = class_weights[2:9]*self.tune
    class_weights[17] = class_weights[17]*5'''

    criterion = nn.CrossEntropyLoss().to(device=device)

    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL'].long())

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
    return {'1_1': data['3D_LABEL']}
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

def Conv3d3x3BN(inc, outc):
    return nn.Sequential(
        nn.Conv3d(inc, outc, 3, padding=1, bias=False),
        nn.BatchNorm3d(outc),
    )

class Decoder(nn.Module):
    def __init__(self, ch_lst, de_ch):
        super(Decoder,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lateral3 = nn.Sequential(
            Conv3d3x3BN(ch_lst[2], de_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            Conv3d3x3BN(de_ch, de_ch))
        self.lateral2 = nn.Sequential(
            Conv3d3x3BN(ch_lst[1], de_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=1, mode='trilinear'),
            Conv3d3x3BN(de_ch, de_ch))
        self.fuse123 = nn.Sequential(
            SELayer(de_ch),
            nn.ReLU(inplace=True),
            Conv3d3x3BN(de_ch, de_ch),
            nn.Upsample(scale_factor=2, mode='trilinear'))
        self.lateral1 = Conv3d3x3BN(ch_lst[0], de_ch)
        self.p3 = nn.Parameter(torch.full([1], 0.8))
        self.p2 = nn.Parameter(torch.full([1], 0.2))
        self.p1 = nn.Parameter(torch.full([1], 0.1))
        self.layer5 = nn.Sequential(
            nn.Conv3d(de_ch, 3, 1, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=3,out_channels=1,kernel_size=1,stride=1,padding=0),
        )

    def forward(self, x_lst):
        data_shape, x1, x2, x3 = x_lst
        x1 = self.lateral1(x1) * self.p1
        x2 = self.lateral2(x2) * self.p2
        x3 = self.lateral3(x3) * self.p3

        x = self.fuse123(x2 + x3) + x1
        x = self.layer5(x)
        x = nn.functional.interpolate(input=x,size=tuple(data_shape[2:]),mode='trilinear')
        return x