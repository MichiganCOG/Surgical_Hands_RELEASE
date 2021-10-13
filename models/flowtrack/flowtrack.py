# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

import datasets.preprocessing_transforms as pt

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

class FlowTrack(nn.Module):

    def __init__(self, **kwargs):

        hands_preprocessing = ['Mixed_Hands', 'Manual_Hands', 'Hand_Data', 'Surgical_Hands', 'Surgical_Hands_v2']

        if kwargs['dataset'] in hands_preprocessing:
            self.train_transforms = PreprocessTrainHand(**kwargs)
            self.test_transforms  = PreprocessEvalHand(**kwargs)
        else:
            self.train_transforms = PreprocessTrainFlowTrack(**kwargs) 
            self.test_transforms  = PreprocessEvalFlowTrack(**kwargs)

        num_layers = kwargs['num_layers']
        block, layers = resnet_spec[num_layers]

        self.inplanes = 64
        self.deconv_with_bias = kwargs['deconv_with_bias']

        super(FlowTrack, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            kwargs['num_deconv_layers'],
            kwargs['num_deconv_filters'],
            kwargs['num_deconv_kernels'],)

        self.final_layer = nn.Conv2d(
            in_channels=kwargs['num_deconv_filters'][-1],
            out_channels=kwargs['num_joints'],
            kernel_size=kwargs['final_conv_kernel'],
            stride=1,
            padding=1 if kwargs['final_conv_kernel'] == 3 else 0
        )

        #for saving and/or outputting visual features
        self.save_feat = kwargs['save_feat']
        self.out_feat  = kwargs['out_feat'] 
        #self.pooling = nn.AdaptiveAvgPool2d((1,1))

        #Also try with max pooling instead 
        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            self.init_weights()

        image_height, image_width = kwargs['final_shape']
        self.heatmap_size = kwargs['heatmap_size']

        self.network_stride  = (image_width/self.heatmap_size[0],
                        image_height/self.heatmap_size[1])#effective stride of the entire network

        self.num_joints = kwargs['num_joints']

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        feats = []
        #flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
        #        [9, 10], [11, 12], [13, 14], [15, 16]]
        flip_pairs = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
        
        #expected shape: [batch,3,T,H,W]
        B = x.shape[0]
        T = x.shape[2]

        test = False 
        for t in range(T):
            output = self.forward_one(x[:,:,t])

            if test:
                input_flipped = np.flip(x[:,:,t].cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = self.forward_one(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                #NOTE: Figure out what this means
                # feature is not aligned, shift flipped heatmap for higher accuracy
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                out = (output + output_flipped) * 0.5
            else:
                out = output 

            if self.save_feat or self.out_feat:
                outputs.append(out[0])
                feats.append(out[1])
            else:
                outputs.append(out)

        if self.save_feat or self.out_feat:
            return {'outputs':torch.stack(outputs, dim=1), 'feat':torch.stack(feats, dim=1)}
        else: 
            return torch.stack(outputs, dim=1)

    def forward_one(self, x):
        #x1_0 = x.cpu()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        vis = self.pooling(x)
        vis = torch.flatten(vis,1)

        x = self.layer1(x)
        #x2_0 = torch.sum(x,dim=1).cpu().numpy()
        x = self.layer2(x)
        #x3_0 = torch.sum(x,dim=1).cpu().numpy()
        x = self.layer3(x)
        #x4_0 = torch.sum(x,dim=1).cpu().numpy()
        x = self.layer4(x)
        #x5_0 = torch.sum(x,dim=1).cpu().numpy()

        x = self.deconv_layers(x)
        #x6_0 = torch.sum(x,dim=1).cpu().numpy()
        x = self.final_layer(x)

        '''
        import matplotlib.pyplot as plt
        plt.subplot(2,4,1)
        plt.title('rgb image')
        plt.imshow(x1_0[0].permute(1,2,0))
        plt.subplot(2,4,2)
        plt.title('Layer 1')
        plt.imshow(x2_0[0])
        plt.colorbar()
        plt.subplot(2,4,3)
        plt.title('Layer 2')
        plt.imshow(x3_0[0])
        plt.colorbar()
        plt.subplot(2,4,4)
        plt.title('Layer 3')
        plt.imshow(x4_0[0])
        plt.colorbar()
        plt.subplot(2,4,5)
        plt.title('Layer 4')
        plt.imshow(x5_0[0])
        plt.colorbar()
        plt.subplot(2,4,6)
        plt.title('Deconv Layer')
        plt.imshow(x6_0[0])
        plt.colorbar()
        plt.subplot(2,4,7)
        plt.title('final output')
        plt.imshow(torch.sum(x[0], dim=0).cpu().numpy())
        plt.colorbar()

        plt.show()
        '''

        if self.save_feat or self.out_feat:
            return x, vis
        else:
            return x

    def init_weights(self, pretrained='./weights/resnet152-b121ed2d.pth'):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('Loaded pretrained model {}'.format(pretrained))

        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

def flip_back(output_flipped, matched_parts):
    ''' 
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4
    'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp 

    return output_flipped

#Source: python-openpose repo; Hzzone
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img 
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1)) 
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1)) 
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1)) 
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1)) 
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

class PreprocessTrainFlowTrack(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """

        self.transforms  = []
        self.preprocess  = kwargs['preprocess']
        crop_type        = kwargs['crop_type']

        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.AffineTransformClip(**kwargs))
        self.transforms.append(pt.RandomFlipClip(direction='h', p=0.5, **kwargs))

        self.transforms.append(pt.ToTensorClip(**kwargs))
        
    def __call__(self, input_data, params):
        center     = params['center']
        scale      = params['scale']
        key_pts    = params['key_pts']

        out_params = {}
        out_params['trans'] = None 
        out_params['inv_trans'] = None 
        out_params['flip'] = False

        for transform in self.transforms:
            if isinstance(transform, pt.AffineTransformClip):
                transform._update_params(center=center, scale=scale)
                out_params['trans'] = transform.trans
                out_params['inv_trans'] = transform.inv_trans

            if key_pts == []:
                input_data = transform(input_data)
            else:
                input_data, key_pts = transform(input_data, key_pts)

            if isinstance(transform, pt.RandomFlipClip):
                out_params['flip'] = transform.flip  

        return input_data, key_pts, out_params  


class PreprocessEvalFlowTrack(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """

        self.transforms = []

        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.AffineTransformClip(test=True,**kwargs))

        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data, params):
        center  = params['center']
        scale   = params['scale']
        key_pts = params['key_pts']

        out_params = {}
        out_params['trans'] = None 
        out_params['inv_trans'] = None 
        out_params['flip'] = False

        for transform in self.transforms:
            if isinstance(transform, pt.AffineTransformClip):
                transform._update_params(center=center, scale=scale)
                out_params['trans'] = transform.trans
                out_params['inv_trans'] = transform.inv_trans

            if key_pts == []:
                input_data = transform(input_data)
            else:
                input_data, key_pts = transform(input_data, key_pts)

        return input_data, key_pts, out_params

class PreprocessTrainHand(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        crop_type = kwargs['crop_type']
        self.transforms = []

        if kwargs['hand_jitter']:
            #Perform this transform first because PIL operations destroy floating point accuracy
            class_kwargs = {'brightness':0.4,'contrast':0.4,'saturation':0.4,'hue':0.4}
            self.transforms.append(pt.ApplyToPIL(transform=torchvision.transforms.ColorJitter, class_kwargs=class_kwargs))

        self.transforms.append(pt.SubtractRGBMean(**kwargs))

        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        elif crop_type=='RandomFrame':
            self.transforms.append(pt.ApplyToClip(transform=torchvision.transforms.RandomCrop(**kwargs)))
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(**kwargs))
        elif crop_type == 'CropClip':
            self.transforms.append(pt.CropClip(**kwargs))

        if kwargs['hand_scale']:
            min_scale = kwargs['hand_scale_amount'][0]
            max_scale = kwargs['hand_scale_amount'][1]
            self.transforms.append(pt.RandomZoomClip(scale=(min_scale, max_scale)))

        if kwargs['hand_rotate']:
            min_deg = kwargs['hand_rotate_amount'][0]
            max_deg = kwargs['hand_rotate_amount'][1]
            self.transforms.append(pt.RandomRotateClip(angles=np.arange(min_deg,max_deg), **kwargs))

        if kwargs['hand_translate']:
            max_tx = kwargs['hand_translate_amount'][0]
            max_ty = kwargs['hand_translate_amount'][1]
            self.transforms.append(pt.RandomTranslateClip(translate=(max_tx, max_ty), **kwargs))

        self.transforms.append(pt.RandomFlipClip(direction='h', p=0.5, **kwargs))

        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data, params):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 
            hand_crop:  Region (around hand) to crop from input image
            label:      Is left hand 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """

        bbox_data = params['bbox_data']
        hand_crop = params['hand_crop']
        label     = params['label']
        angle     = params.get('in_rot', None)

        out_params = {}
        out_params['flip'] = False

        for transform in self.transforms:
            if isinstance(transform, pt.CropClip):
                transform._update_bbox(hand_crop[0], hand_crop[2], hand_crop[1], hand_crop[3], True)

            if isinstance(transform, pt.RandomRotateClip) and angle is not None:
                transform._update_angles([angle])

            input_data, bbox_data = transform(input_data, bbox_data)

            if isinstance(transform, pt.RandomFlipClip):
                out_params['flip'] = transform.flip  

            if isinstance(transform, pt.RandomRotateClip):
                out_params['out_rot'] = transform.out_rot
        
        return input_data, bbox_data, out_params

class PreprocessEvalHand(object):
    """
    Container for all transforms used to preprocess clips for evaluation in this dataset.
    """
    def __init__(self, **kwargs):
        crop_type = kwargs['crop_type']
        self.transforms = []

        self.transforms.append(pt.SubtractRGBMean(**kwargs))

        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        elif crop_type=='RandomFrame':
            self.transforms.append(pt.ApplyToClip(transform=torchvision.transforms.RandomCrop(**kwargs)))
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(**kwargs))
        elif crop_type == 'CropClip':
            self.transforms.append(pt.CropClip(**kwargs))

        #self.transforms.append(pt.RandomFlipClip(direction='h', p=1.0, **kwargs))
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.ToTensorClip())


    def __call__(self, input_data, params):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 
            hand_crop:  Region (around hand) to crop from input image
            label:      Is left hand 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """

        bbox_data = params['bbox_data']
        hand_crop = params['hand_crop']
        label     = params['label']

        out_params = {}
        out_params['flip'] = False

        for transform in self.transforms:
            if isinstance(transform, pt.CropClip):
                transform._update_bbox(hand_crop[0], hand_crop[2], hand_crop[1], hand_crop[3], True)
            input_data, bbox_data = transform(input_data, bbox_data)


        return input_data, bbox_data, out_params 

