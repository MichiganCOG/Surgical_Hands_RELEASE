import numpy as np
from scipy import ndimage
import os
import cv2

import torch 
import torch.nn as nn
import torch.nn.functional as F


class Losses(object):
    def __init__(self, *args, **kwargs): #loss_type, size_average=None, reduce=None, reduction='mean', *args, **kwargs):
        """
        Class used to initialize and handle all available loss types in ViP

        Args: 
            loss_type (String): String indicating which custom loss function is to be loaded.

        Return:
            Loss object 
        """

        self.loss_type   = kwargs['loss_type']
        self.loss_object = None
        
        if self.loss_type == 'MSE':
            self.loss_object = MSE(*args, **kwargs)

        elif self.loss_type == 'Hand_Heatmap_MSE':
            self.loss_object = Hand_Heatmap_MSE(*args, **kwargs)

        elif self.loss_type == 'M_XENTROPY':
            self.loss_object = M_XENTROPY(*args, **kwargs)

        elif self.loss_type == 'JointsMSELoss':
            self.loss_object = JointsMSELoss(*args, **kwargs)
        
        elif self.loss_type == 'ContrastiveLoss':
            self.loss_object = ContrastiveLoss(*args, **kwargs)

        else:
            print('Invalid loss type selected. Quitting!')
            exit(1)

    def loss(self, predictions, data, **kwargs):
        """
        Function that calculates loss from selected loss type

        Args:
            predictions (Tensor, shape [N,*]): Tensor output by the network
            target      (Tensor, shape [N,*]): Target tensor used with predictions to compute the loss

        Returns:
            Calculated loss value
        """ 
        return self.loss_object.loss(predictions, data, **kwargs)

class MSE():
    def __init__(self, *args, **kwargs):
        """
        Mean squared error (squared L2 norm) between predictions and target

        Args:
            reduction (String): 'none', 'mean', 'sum' (see PyTorch Docs). Default: 'mean'
            device    (String): 'cpu' or 'cuda'

        Returns:
            None 
        """

        reduction = 'mean' if 'reduction' not in kwargs else kwargs['reduction']
        self.device = kwargs['device']

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data

        Returns:
            Return mean squared error loss
        """

        targets = data['labels'].to(self.device)

        return self.mse_loss(predictions, targets)

class Hand_Heatmap_MSE():
    def __init__(self, *args, **kwargs):
        """
        Mean squared error (squared L2 norm) between predictions and target

        Args:
            reduction (String): 'none', 'mean', 'sum' (see PyTorch Docs). Default: 'mean'
            device    (String): 'cpu' or 'cuda'

        Returns:
            None 
        """

        reduction = 'mean' if 'reduction' not in kwargs else kwargs['reduction']
        self.device = kwargs['device']

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor): Output by the network
            data         (dictionary)
                - labels (Tensor):  Targets from ground truth data
        """

        out1, out2, out3, out4, out5, out6 = predictions
        targets = data['heatmaps'].to(self.device)
    
        mask = 1 - data['occ'].to(self.device)[:,0,:,None,None].float()
        targets = targets[:,0] * mask
        
        '''
        import matplotlib.pyplot as plt

        vid_data = data['data'][0].permute(1,2,0)
        heatmap = predictions[-1][0]
        plt.figure(figsize=(16,10))
        extent = np.int(0), np.int(368), np.int(0), np.int(368)

        for i in range(22):
            plt.subplot(5,5,i+1)
            plt.imshow(vid_data, interpolation='none', extent=extent)
            plt.imshow(heatmap[i,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
            plt.colorbar()

        plt.subplot(5,5,23)
        plt.imshow(vid_data, interpolation='none', extent=extent)
        plt.imshow(targets[0,-1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
        plt.colorbar()

        plt.subplot(5,5,24)
        plt.imshow(vid_data, interpolation='none', extent=extent)
        plt.imshow(targets[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
        plt.colorbar()
        
        plt.show()
        '''

        #loss includes many intermediate supervision steps
        total_loss = self.mse_loss(out1 * mask, targets) + self.mse_loss(out2 * mask, targets) + \
               self.mse_loss(out3 * mask, targets) + self.mse_loss(out4 * mask, targets) + \
               self.mse_loss(out5 * mask, targets) + self.mse_loss(out6 * mask, targets) 
        
        heat_weight = 46*46*22
        return total_loss * heat_weight
    
class M_XENTROPY(object):
    def __init__(self, *args, **kwargs):
        """
        Cross-entropy Loss with a distribution of values, not just 1-hot vectors 

        Args:
            dim (integer): Dimension to reduce 

        Returns:
            None 
        """
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data
                
        Return:
            Cross-entropy loss  
        """

        targets = data['labels']
        one_hot = np.zeros((targets.shape[0], predictions.shape[1]))
        one_hot[np.arange(targets.shape[0]), targets.cpu().numpy().astype('int32')[:, -1]] = 1
        one_hot = torch.Tensor(one_hot).cuda()

        return torch.mean(torch.sum(-one_hot * self.logsoftmax(predictions), dim=1))

#https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/loss.py
class JointsMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(JointsMSELoss, self).__init__()

        self.criterion = nn.MSELoss()

        self.device    = kwargs['device']
        self.use_target_weight = kwargs['use_target_weight']
        
        self.heatmap_size   = kwargs['heatmap_size']
        self.loss_weight = np.prod(self.heatmap_size) #scale loss by this value 

    def loss(self, predictions, data):
        target = data['heatmaps'].to(self.device)
        target_weight = data['heatmap_weights'].float().to(self.device)
	
        B, O, num_joints, H, W = predictions.shape
        predictions = predictions.reshape(B*O, num_joints, H, W)
        target      = target.reshape(B*O, num_joints, H, W)
        target_weight = target_weight.reshape(B*O, num_joints, -1)

        batch_size = predictions.shape[0]

        heatmaps_pred = predictions.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0 

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints   

#Code source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, *args, **kwargs):
        super(ContrastiveLoss, self).__init__()

        self.margin = kwargs['cont_loss_margin']
        self.eps = 1e-9

        self.device = kwargs['device']

    def loss(self, predictions, data):
        """
        Args:
            predictions (Tuple):
                - output1 (Tensor, shape [N, D]) 
                - output2 (Tensor, shape [N, D])
            data (Dictionary):
                - pair_label (Tensor, shape [N, 1])
        """

        output1, output2 = predictions
        target = data['pair_label'].to(self.device)

        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        return losses.mean()
