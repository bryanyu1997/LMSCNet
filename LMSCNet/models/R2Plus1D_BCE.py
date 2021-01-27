import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as tf


class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in


class R2Plus1D_BCE(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    f = self.input_dimensions[1]

    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

    self.encoder = models.video.r2plus1d_18(pretrained=True)
    self.encoder.stem[0] = self.change_channel(self.encoder.stem[0])
    self.z_downsample = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(17, 1, 1), stride=(2,1,1), padding=(8, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),)
    self.decoder = Decoder()

    '''self.Encoder_block1 = nn.Sequential(
      nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block2 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block3 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block4 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    # Treatment output 1:8
    self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
    self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
    self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

    # Treatment output 1:4
    self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
    self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
    self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

    # Treatment output 1:2
    self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
    self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

    # Treatment output 1:1
    self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
    self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
    self.seg_head_1_1       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])'''

  def forward(self, x):

    inputs = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    inputs = torch.squeeze(inputs, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]
    inputs = inputs.unsqueeze(1)

    feature_lst = []
    feature_lst.append(inputs.shape)
    x1 = self.encoder.stem(inputs)
    x2 = self.z_downsample(x1)
    x = self.encoder.layer1(x2)
    feature_lst.append(x)
    x = self.encoder.layer2(x)
    feature_lst.append(x)
    x = self.encoder.layer3(x)
    feature_lst.append(x)
    x = self.encoder.layer4(x)
    feature_lst.append(x)
    output = self.decoder(feature_lst)

    # Take back to [W, H, D] axis order
    if self.training:
      out_scale_1_1__3D = output.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    else:
      out_scale_1_1__3D = output.permute(0, 1, 3, 2, 4)
      out_scale_1_1__3D[:,:1] = nn.Sigmoid()(out_scale_1_1__3D[:,:1])
      out_scale_1_1__3D[:,1:] = nn.Softmax(dim=1)(out_scale_1_1__3D[:,1:])
      out_scale_1_1__3D[:,1:] = (1-out_scale_1_1__3D[:,:1])* out_scale_1_1__3D[:,1:]
    
    scores = {'pred_semantic_1_1': out_scale_1_1__3D}

    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''
    target = data['3D_LABEL']['1_1'].clone()
    scores = scores['pred_semantic_1_1'].clone()
    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=device, dtype=dtype)

    mask_bce = (target!=255)
    target_bce = (target==0).float()[mask_bce]
    scores_bce = scores[:,0].clone()[mask_bce]
    class_weights_bce = class_weights[:1].clone()

    #mask_ce = (target!=0) & (target!=255)
    target_ce = (target-1).clone()
    target_ce[(target_ce==-1)|(target_ce==254)] = 255
    scores_ce = scores[:,1:].clone()
    class_weights_ce = class_weights[1:].clone()

    criterion_1 = nn.BCEWithLogitsLoss(pos_weight=class_weights_bce, reduction='mean').to(device=device)
    #criterion_2 = nn.CrossEntropyLoss(weight=class_weights_ce, ignore_index=255, reduction='mean').to(device=device)

    loss_bce = criterion_1(scores_bce, target_bce)
    #loss_ce = criterion_2(scores_ce, target_ce.long())

    #loss_1_1 = loss_bce + loss_ce
    loss_1_1 = loss_bce * 10
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

def Conv3d3x3BNReLU(inc, outc):
    return nn.Sequential(
        nn.Conv3d(inc, outc, 3, padding=1, bias=False),
        nn.BatchNorm3d(outc),
        nn.ReLU(inplace=True),
    )

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
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
        data_shape, x1, x2, x3, x4 = x_lst
        x = self.layer1(x4)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral3(x3)
        x = self.layer2(x)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral2(x2)
        x = self.layer3(x)
        x = nn.Upsample(scale_factor=(2,2,2))(x) + self.lateral1(x1)
        x = self.layer4(x)
        x = self.layer_5(x)
        x = nn.functional.interpolate(input=x,size=tuple(data_shape[2:]),mode='trilinear')

        return x