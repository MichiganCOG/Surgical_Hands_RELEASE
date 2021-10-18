# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

#Concatenates prior heatmap and visual features, then passes through attention mechanism
#Output is some weighted prior heatmap concatenated with current heatmap output
#Weight prior heatmap and current heatmap are concanetated and passed further through layers

#This model integrates heatmap predictions on the prior frames as well  
#Here we take the prior prediction exactly as it is. We don't create a new heatmap based on
#the maximum points

#Additionally, we use a linear scheduler to transition between GT priors and prediction prior

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

import datasets.preprocessing_transforms as pt

import cv2

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, extra_layers=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes+extra_layers, planes, kernel_size=1, bias=False)
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

class FlowTrack_R_GT_V5_Linear(nn.Module):

    def __init__(self, **kwargs):

        hands_preprocessing = ['Mixed_Hands', 'Manual_Hands', 'Hand_Dets', 'Surgical_Hands', 'Surgical_Hands_v2']

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
        self.hm_to_layer = kwargs['hm_to_layer']
        self.num_joints  = kwargs['num_joints']
        self.image_height, self.image_width = kwargs['final_shape']
        self.heatmap_size  = kwargs['heatmap_size']
        self.stride        = (self.image_width/self.heatmap_size[0],
                self.image_height/self.heatmap_size[1])#effective stride of the entire network

        self.prior_threshold      = kwargs['prior_threshold']
        self.min_gauss_peak_train = kwargs['min_gauss_peak_train']
        self.epoch = kwargs['epoch'] #default to final epoch (in case running eval.py only - training should reset to 0)

        self.total_priors = 0
        self.use_gt       = 0 #or no priors exist
        self.use_pred     = 0

        super(FlowTrack_R_GT_V5_Linear, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 

        self.sigmoid = nn.Sigmoid()
        self.prior_attention = nn.Sequential(
                                nn.Conv2d(64 + self.num_joints, 256, kernel_size=3, padding=1, stride=1), #64 channels + num joints
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(256, self.num_joints, kernel_size=4, padding=1, stride=2),
                                nn.ReLU()
                                )

        self.prior_update = nn.Sequential( #use (weighted?) prior to update current output
                                nn.Conv2d(2*self.num_joints, 256, kernel_size=3, padding=1, stride=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(),
                                nn.ConvTranspose2d(256, self.num_joints, kernel_size=4, padding=1, stride=2),
                                )

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            kwargs['num_deconv_layers'],
            kwargs['num_deconv_filters'],
            kwargs['num_deconv_kernels'],)

        self.final_layer = nn.Conv2d(
            in_channels=kwargs['num_deconv_filters'][-1],
            out_channels=self.num_joints,
            kernel_size=kwargs['final_conv_kernel'],
            stride=1,
            padding=1 if kwargs['final_conv_kernel'] == 3 else 0
        )

        #for saving and/or outputting visual features
        self.save_feat = kwargs['save_feat']
        self.out_feat  = kwargs.get('out_feat', False)

        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            self.init_weights()

    def reset_vals(self):
        self.total_priors = 0
        self.use_gt       = 0
        self.use_pred     = 0

    def update_epoch(self, epoch):
        self.epoch = epoch

    def _make_layer(self, block, planes, blocks, stride=1, append_hm=False):
        downsample = None
        extra_layers = 17 if append_hm else 0

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes + extra_layers, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, extra_layers))
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

    def forward(self, x, gt_heatmap=None, x_prev=None, params=None):
        outputs = []
        feats   = []
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
        
        #expected shape x: [batch,3,T,H,W]
        #      gt_heatmap: [batch,T,J,H*,W*]
        B = x.shape[0]
        T = x.shape[2]
        test = False 
        t0_list = list(range(T-1))
        t0_list.insert(0,0)
        for t0,t1 in zip(t0_list, range(T)):
            if x_prev is not None:
                with torch.no_grad():
                    hand_crops  = params['input_crop']
                    prior_crops = params['prior_crop']
                    frame_sizes = params['frame_size']
                    load_type   = params['load_type'][0] #expected train or val
                    padding     = params.get('padding', None)
                    prior_pads  = params.get('prior_pad', None)
                    strides     = params.get('stride', None)
                    inv_trans   = params.get('inv_trans', None)
                    trans       = params.get('trans', None)
                    flipped     = params.get('flipped', None)

                    output_prev = self.forward_one(x_prev[:,:,t0], None)

                    prev_heatmap = []
                    keep_priors = []
                    self.total_priors += B
                    use_pred_prior_prob = min(0.10 * self.epoch, 1) #probability of using prediction prior
                    for i, hm in enumerate(output_prev):
                        if  np.random.rand() > use_pred_prior_prob and load_type != 'val':
                            self.use_gt += 1
                            continue 
                        
                        self.use_pred += 1
                        img_width, img_height = frame_sizes[i]
                        
                        if prior_crops[i].tolist() != [-1,-1,-1,-1]: #No prior to adjust
                            #re-transform for next image crop
                            x1,y1,x2,y2 = prior_crops[i].tolist()
                            crop_h = y2 - y1
                            crop_w = x2 - x1

                            temp0 = hm.cpu()
                            ######### For human pose
                            if not strides is None:
                                hm_ = hm.permute(1,2,0).cpu().numpy()

                                if flipped[i]:
                                    hm_ = cv2.flip(hm_, 1)
                                hm_ = cv2.resize(hm_, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
                                temp1 = hm_

                                hm_ = cv2.warpAffine(hm_, inv_trans[i,0].cpu().numpy(), (img_width.item(), img_height.item()), flags=cv2.INTER_LINEAR)
                                temp2 = hm_
                                hm_ = cv2.warpAffine(hm_, trans[i,0].cpu().numpy(), dsize=(self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
                                hm_ = cv2.resize(hm_, tuple(self.heatmap_size), interpolation=cv2.INTER_CUBIC)
                                temp3 = hm_

                                if flipped[i]:
                                    hm_ = cv2.flip(hm_, 1)
                                hm_ = torch.from_numpy(hm_).permute(2,0,1).to(x.device)

                                '''
                                import matplotlib.pyplot as plt

                                plt.subplot(2,4,1)
                                mean = np.array([[[123.675,116.28,103.52]]])
                                std = np.array([[[58.395,57.12,57.375]]])
                                aux_vis = np.clip(((x_prev[i,:,0].permute(1,2,0).cpu().numpy()*std)+mean)/255,0,1)
                                plt.imshow(aux_vis)

                                plt.subplot(2,4,2)
                                plt.imshow(torch.max(temp0, dim=0)[0].detach().numpy())
                                plt.title('Network output')
                                plt.subplot(2,4,3)
                                plt.imshow(np.max(temp1, axis=-1))
                                plt.title('Re-interpolated output')
                                plt.subplot(2,4,4)
                                plt.imshow(np.max(temp2, axis=-1))
                                plt.title('Reprojected to full frame')
                                plt.subplot(2,4,5)
                                plt.imshow(np.max(temp3, axis=-1))
                                plt.title('t1 resized ')

                                plt.show()
                                '''
                            else:
                                ######### For hand pose
                                hm_ = F.interpolate(hm[None], size=(crop_h,crop_w))
                                temp1 = hm_[0].cpu()

                                pl,pt,pr,pb = prior_pads[i].tolist()

                                pad_tensor = nn.ConstantPad2d((x1, max(0, ((img_width+pl+pr)-x2)), y1, max(0, ((img_height+pt+pb)-y2))), 0.0)  #pad_left, pad_right, pad_top, pad_bot
                                hm_ = pad_tensor(hm_) #prior hand crop reprojected onto full frame
                                temp2 = hm_[0].cpu()

                                _pb = (img_height+pt) if not pb else -pb #check if non-zero, and adjust for array slicing
                                _pr = (img_width+pl) if not pr else -pr

                                hm_ = hm_[0][:,pt:_pb,pl:_pr] #prior hand crop w/o padding
                                temp3 = hm_.cpu()

                                pl,pt,pr,pb = padding[i,0].tolist() #current image padding
                                x1,y1,x2,y2 = hand_crops[i,0].tolist() #current image hand crop
                                #add current image padding
                                pad_tensor = nn.ConstantPad2d((pl,pr,pt,pb), 0.0)  #pad_left, pad_right, pad_top, pad_bot
                                hm_ = pad_tensor(hm_) #current hand crop w/ padding (only right and bottom padding need to be added)
                                temp4 = hm_.cpu()
                                hm_ = hm_[:,int(y1):int(y2),int(x1):int(x2)] #current crop position
                                temp5 = hm_.cpu()
                                hm_ = F.interpolate(hm_[:,None], size=self.heatmap_size)[:,0] #resized to heatmap size
                                temp6 = hm_.cpu()

                                '''
                                import matplotlib.pyplot as plt

                                plt.subplot(2,4,1)
                                plt.imshow(torch.max(temp0, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('Network output')
                                plt.subplot(2,4,2)
                                plt.imshow(torch.max(temp1, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('Re-interpolated output')
                                plt.subplot(2,4,3)
                                plt.imshow(torch.max(temp2, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('Reprojected to full frame')
                                plt.subplot(2,4,4)
                                plt.imshow(torch.max(temp3, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('Removed padding')

                                plt.subplot(2,4,6)
                                plt.imshow(torch.max(temp4, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('full image with padding')
                                plt.subplot(2,4,7)
                                plt.imshow(torch.max(temp5, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('t1 hand crop')
                                plt.subplot(2,4,8)
                                plt.imshow(torch.max(temp6, dim=0)[0].detach().numpy(),vmin=0, vmax=1)
                                plt.title('t1 resized to heatmap size')
                                plt.show()
                                '''
                        else:
                            hm_ = hm

                        prev_heatmap.append(hm_)
                        keep_priors.append(i)

                    #Replace corresponding minibatch prior with minibatch in gt_heatmap
                    if len(prev_heatmap) > 0:
                        prev_heatmap = torch.stack(prev_heatmap).to(x.device)
                        gt_heatmap[keep_priors,t0] = prev_heatmap 

            if gt_heatmap is not None:
                output = self.forward_one(x[:,:,t1], gt_heatmap[:,t0])
            else:
                output = self.forward_one(x[:,:,t1], None)

            if test:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
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
            return {'outputs':torch.stack(outputs, dim=1), 'feat':torch.stack(feats,dim=1)}
        else:
            return torch.stack(outputs, dim=1)

    def forward_one(self, x1, x0, params=None):
        B,C,H,W = x1.shape 
        H_ = self.heatmap_size[1] 
        W_ = self.heatmap_size[0]

        if x0 is None:
            x0 = torch.zeros(B,self.num_joints,H_,W_).to(x1.device)

        x1_0 = x1.cpu()

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        vis = x1.clone() #(B,64,H_,W_)
        if vis.shape[2] != H_ or vis.shape[3] != W_:
            vis = F.interpolate(vis, size=(H_,W_))
        
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.deconv_layers(x1)
        x1 = self.final_layer(x1)

        #Add 1x1 convolution + non-linearity  
        x0_0 = torch.max(x0,dim=1)[0].cpu().numpy()
        x0 = torch.cat((vis,x0), dim=1)
        x0 = self.prior_attention(x0)
        x0_1 = torch.max(x0,dim=1)[0].cpu().detach().numpy()

        x1_1 = torch.max(x1,dim=1)[0].cpu().detach().numpy()
        x1 = torch.cat((x1,x0), dim=1)
        x1 = self.prior_update(x1)

        '''
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(18, 10))

        rr = len(x1)
        for ii in range(rr):
            plt.subplot(rr,5,(5*ii) + 1)
            plt.title('heatmap prior t-1')
            plt.imshow(x0_0[ii], vmin=0, vmax=1)
            #plt.colorbar()

            plt.subplot(rr,5,(5*ii) + 2)
            plt.title('cropped image')
            extent = np.int(0), np.int(W), np.int(0), np.int(H)
            mean = np.array([[[123.675,116.28,103.52]]])
            std  = np.array([[[58.395,57.12,57.375]]])
            _vis  = np.clip(((x1_0[ii].permute(1,2,0).numpy()*std)+mean)/255,0,1)
            plt.imshow(_vis, interpolation='none', extent=extent)
            #plt.imshow(x0_0[ii], cmap='jet', alpha=0.5, interpolation='none', extent=extent)

            plt.subplot(rr,5,(5*ii) + 3)
            plt.title('modified heatmap t-1')
            plt.imshow(x0_1[ii])
            #plt.colorbar()

            plt.subplot(rr,5,(5*ii) + 4)
            plt.title('heatmap t')
            plt.imshow(x1_1[ii])
            #plt.colorbar()

            plt.subplot(rr,5,(5*ii) + 5)
            plt.title('final heatmap t')
            plt.imshow(_vis, interpolation='none', extent=extent)
            plt.imshow(torch.max(x1[ii], dim=0)[0].cpu().detach().numpy(),vmin=0, vmax=1, cmap='jet', alpha=0.25, interpolation='none', extent=extent)
            #plt.colorbar()
        plt.show()
        #os.makedirs('./eval_vis_outputs', exist_ok=True)
        #filename = str(params['frame_id'])+'_'+str(params['batch_num'])+'.png'
        #plt.savefig('./eval_vis_outputs/'+filename)
        #plt.close()
        '''

        #if self.save_feat or self.out_feat:
        #    vis = self.pooling(vis)
        #    vis = torch.flatten(vis,1)
        #    return x1, vis 
        #else:
        return x1

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

            #NOTE: Some changes to loading ImageNet weights
            layers_to_remove = ['layer1.0.conv1.weight','layer1.0.downsample.0.weight'] #weights won't be the same b/c of my changes
            for l in layers_to_remove:
                del pretrained_state_dict[l]

            model_dict = self.state_dict()
            model_dict.update(pretrained_state_dict)

            self.load_state_dict(model_dict, strict=False)

        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

def get_max_preds(batch_heatmaps):
    '''  
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals

#Adapted from: https://github.com/microsoft/human-pose-estimation.pytorch
def generate_target(joints, num_keypoints, heatmap_size, stride, min_gauss_peak_train):
    ''' 
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    
    #num_keypoints = kwargs['num_keypoints']
    #heatmap_size  = kwargs['heatmap_size']
    sigma         = 3.0
    #stride        = kwargs['stride']

    target_weight = np.ones((num_keypoints, 1), dtype=np.float32)
    target_weight[:, 0] = joints[:, -1]

    target = np.zeros((num_keypoints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_keypoints):
        mu_x = int(joints[joint_id][0] / stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > min_gauss_peak_train:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def crop_coords(x, y, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        if np.any(x > crop_xmax) or np.any(x < crop_xmin) or np.any(y > crop_ymax) or np.any(y < crop_ymin):
            return -1*np.ones(x.shape), -1*np.ones(y.shape)

        x_new = np.clip(x, crop_xmin, crop_xmax)
        y_new = np.clip(y, crop_ymin, crop_ymax)

        return x_new-crop_xmin, y_new-crop_ymin

def resize_pt_coords(x, y, img_shape, resize_shape):
        # Get relative position for point coords within a frame, after it's resized

        img_h = img_shape[0]
        img_w = img_shape[1]

        res_h = resize_shape[0]
        res_w = resize_shape[1]

        frac_h = res_h/float(img_h)
        frac_w = res_w/float(img_w)

        x_new = (x * frac_w).astype(int)
        y_new = (y * frac_h).astype(int)

        return x_new, y_new

class PreprocessTrainFlowTrack(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
                transform._update_bbox(bbox_data[0], bbox_data[2], bbox_data[1], bbox_data[3], True)
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
        in_rot     = params.get('in_rot', None)

        out_params = {}
        out_params['trans'] = None
        out_params['inv_trans'] = None
        out_params['flip'] = False
        out_params['out_rot'] = None 

        for transform in self.transforms:
            if isinstance(transform, pt.AffineTransformClip):
                transform._update_params(center=center, scale=scale, in_rot=in_rot)
                out_params['inv_trans'] = transform.inv_trans
                out_params['trans'] = transform.trans

            if key_pts == []:
                input_data = transform(input_data)
            else:
                input_data, key_pts = transform(input_data, key_pts)

            if isinstance(transform, pt.RandomFlipClip):
                out_params['flip'] = transform.flip
            if isinstance(transform, pt.AffineTransformClip):
                out_params['out_rot'] = transform.out_rot

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
        in_rot     = params.get('in_rot', None)

        out_params = {}
        out_params['trans'] = None
        out_params['inv_trans'] = None 
        out_params['flip'] = False
        out_params['out_rot'] = None 

        for transform in self.transforms:
            if isinstance(transform, pt.AffineTransformClip):
                transform._update_params(center=center, scale=scale, in_rot=in_rot)
                out_params['inv_trans'] = transform.inv_trans
                out_params['trans'] = transform.trans

            if key_pts == []:
                input_data = transform(input_data)
            else:
                input_data, key_pts = transform(input_data, key_pts)

            if isinstance(transform, pt.AffineTransformClip):
                out_params['out_rot'] = transform.out_rot

        return input_data, key_pts, out_params

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
            self.default_angles = np.arange(min_deg,max_deg)
            self.transforms.append(pt.RandomRotateClip(angles=self.default_angles, **kwargs))

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

            if isinstance(transform, pt.RandomRotateClip):
                if angle is None: #Ensure full angle selection is reset
                    transform._update_angles(self.default_angles)
                else: #Force selection of input angle
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
        out_params['out_rot'] = None 

        for transform in self.transforms:
            if isinstance(transform, pt.CropClip):
                transform._update_bbox(hand_crop[0], hand_crop[2], hand_crop[1], hand_crop[3], True)
            input_data, bbox_data = transform(input_data, bbox_data)

            if isinstance(transform, pt.RandomRotateClip):
                out_params['out_rot'] = transform.out_rot 

        return input_data, bbox_data, out_params 
