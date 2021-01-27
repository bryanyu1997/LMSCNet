import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as tf
import pdb


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


class R2Plus1D_SPADE(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies, SPADE_hidden_layer=0, SPADE_output_layer=0):
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
    self.spade = SPADE(SPADE_output_layer, SPADE_output_layer, SPADE_hidden_layer)
    '''self.semantic = nn.Sequential(
            nn.Conv3d(in_channels=SPADE_output_layer, out_channels=20, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),)'''

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

    output_res = self.decoder(feature_lst)
    output_spade = self.spade(output_res)
    output = output_res + output_spade

    output = nn.functional.interpolate(input=output,size=tuple(inputs.shape[2:]),mode='trilinear')
    #output = self.semantic(output)

    # Take back to [W, H, D] axis order
    out_scale_1_1__3D = output.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

    scores = {'pred_semantic_1_1': out_scale_1_1__3D}

    return scores

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

    target = data['3D_LABEL']['1_1']
    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    # Weighted-Tuning
    #class_weights[2:9] = class_weights[2:9]*10
    #class_weights[17] = class_weights[17]*5

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)

    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL']['1_1'].long())

    loss = {'total': loss_1_1, 'semantic_1_1': loss_1_1}
    return loss

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

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
        #x = nn.functional.interpolate(input=x,size=tuple(data_shape[2:]),mode='trilinear')

        return x

class SPADE(nn.Module):
    def __init__(self, output_nc, input_nc, nhidden=0):
        super().__init__()

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.norm = nn.BatchNorm2d(input_nc)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(input_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, output_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, output_nc, kernel_size=3, padding=1)

    def forward(self, x):

        #(B,C,D,H,W) -> (D,B,C,H,W)
        x_batch = x.permute(2,0,1,3,4)
        pre_slice = self.norm(x_batch[0].detach()).unsqueeze(0)
        #out_batch = torch.zeros(((0,)+tuple(x.shape[1:])),device=x.device)
        
        for curr_slice in x_batch[1:]:

          # Part 1. generate parameter-free normalized activations
          normalized = self.norm(curr_slice)

          # Part 2. produce scaling and bias conditioned on semantic map
          #segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
          actv = self.mlp_shared(pre_slice[-1].detach())
          actv = nn.ReLU(inplace=True)(actv)
          gamma = self.mlp_gamma(actv)
          beta = self.mlp_beta(actv)

          # apply scale and bias
          out_slice = normalized * (1 + gamma) + beta
          pre_slice = torch.cat((pre_slice, self.norm(out_slice).unsqueeze(0)),dim=0)

        pre_slice = nn.ReLU(inplace=True)(pre_slice)
        output = pre_slice.permute(1,2,0,3,4)

        return output