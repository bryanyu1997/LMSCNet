import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as tf
from LMSCNet.common.transformer_official import Transformer
import pdb

class R32D_linear(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies, downsample=False, 
                 hidden_dim=32, dropout=0.1, 
                 nheads=4, num_encoder_layer=2, num_decoder_layer=2):
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
    self.hidden_dim = hidden_dim

    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]
    
    ''' Encoder Init '''
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
    self.trans_4 = Conv3d3x3BNReLU(256,512)
    self.trans_3 = Conv3d3x3BNReLU(64,256)
    self.trans_2 = Conv3d3x3BNReLU(16,128)
    self.trans_1 = Conv3d3x3BNReLU(4,64)
    self.in_proj = conv2dbnrelu(64,32)
    self.decoder = Decoder()

  def forward(self, x):
    occ = x['3D_OCCLUDED']
    inputs = torch.cat((x['3D_OCCUPANCY'],x['3D_OCCLUDED'].unsqueeze(1)), dim=1)  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    inputs = inputs.permute(0, 1, 3, 2, 4)
    B, C, D, H, W = inputs.shape 
    device = inputs.device
    layer_lst = []
    layer_lst.append(inputs.shape)

    # --------------------------Encoder---------------------------------
    x0 = inputs.reshape(B,-1,H,W)
    x0 = self.stem(x0)
    x1 = self.encoder.layer1(x0)
    x2 = self.encoder.layer2(x1)
    x3 = self.encoder.layer3(x2)
    x4 = self.encoder.layer4(x3)
    # ------------------------Transformer-------------------------------
    x_lat4 = x4.reshape(x4.shape[0],x4.shape[1]//2,-1,x4.shape[-2],x4.shape[-1])
    layer_lst.append(self.trans_4(x_lat4))
    x_lat3 = x3.reshape(x3.shape[0],x3.shape[1]//4,-1,x3.shape[-2],x3.shape[-1])
    layer_lst.append(self.trans_3(x_lat3))
    x_lat2 = x2.reshape(x2.shape[0],x2.shape[1]//8,-1,x2.shape[-2],x2.shape[-1])
    layer_lst.append(self.trans_2(x_lat2))
    x_lat1 = x1.reshape(x1.shape[0],x1.shape[1]//16,-1,x1.shape[-2],x1.shape[-1])
    layer_lst.append(self.trans_1(x_lat1))
    # --------------------------Decoder---------------------------------
    output = self.decoder(layer_lst)

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


def tf_layer(tB, hidden_dim, tH, tW, tD):
    # position encoding (DxHxW,B,hidden_dimx3)
    pos = torch.cat([tf_parameter(tW, hidden_dim).unsqueeze(0).repeat(tH, 1, 1), tf_parameter(tH, hidden_dim).unsqueeze(1).repeat(1, tW, 1)], dim=-1).flatten(0, 1).unsqueeze(dim=0)
    # Input query (DxHxW,B,hidden_dimx3)
    query_emb = tf_query_emb(tD, hidden_dim).weight.unsqueeze(1).repeat(1,tH*tW,1)
    return pos, query_emb

def tf_parameter(i, hidden_dim):
    return nn.Parameter(torch.rand(i, hidden_dim//2))

def tf_query_emb(num_queries, hidden_dim):
    return nn.Embedding(num_queries, hidden_dim)

def conv2dbnrelu(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )

def Conv3d3x3BNReLU(inc, outc):
    return nn.Sequential(
        nn.Conv3d(inc, outc, 3, padding=1, bias=False),
        nn.BatchNorm3d(outc),
        nn.ReLU(inplace=True),
    )
def TransConv3d3x3BNReLU(inc, outc):
    return nn.Sequential(
        nn.ConvTranspose3d(inc, outc, 3, padding=1, bias=False),
        nn.BatchNorm3d(outc),
        nn.ReLU(inplace=True),
    )

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.stem = stem()
        self.lateral3 = Conv3d3x3BNReLU(256,256)
        self.lateral2 = Conv3d3x3BNReLU(128,128)
        self.lateral1 = Conv3d3x3BNReLU(64,64)
        self.layer1 = Conv3d3x3BNReLU(512, 256)
        self.layer2 = Conv3d3x3BNReLU(256, 128)
        self.layer3 = Conv3d3x3BNReLU(128, 64)
        self.layer4 = Conv3d3x3BNReLU(64, 64)
        self.layer_5 = nn.Conv3d(in_channels=64,out_channels=20,kernel_size=1,stride=1,padding=0)


    def forward(self, x_lst):
        data_shape, x4, x3, x2, x1= x_lst
        x = self.layer1(x4)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral3(x3)
        x = self.layer2(x)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral2(x2)
        x = self.layer3(x)
        #x = nn.Upsample(scale_factor=(2,2,2))(x)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral1(x1)
        x = self.layer4(x)
        x = self.layer_5(x)
        x = nn.functional.interpolate(input=x,size=tuple(data_shape[2:]),mode='trilinear')

        return x