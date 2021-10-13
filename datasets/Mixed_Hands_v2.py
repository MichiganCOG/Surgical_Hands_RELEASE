#See if we can train our recursive model using the same frame as an auxiliary input
#Will output the current object and a "previous image crop"
#Will need to apply a few more augmentations during training to prevent overfitting

import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class Mixed_Hands_v2(DetectionDataset):
    """
    Mixture of Manual_Hands and Synthetic_Hands datasets. Test split only contains
    Manual_Hands data 

    16,173 training samples
    843 testing samples
    Source: https://arxiv.org/1704.07809
    """
    def __init__(self, *args, **kwargs):
        super(Mixed_Hands_v2, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']
        self.json_path = kwargs['json_path']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects   = 1
        self.sigma         = kwargs['gaussian_sigma']
        self.heatmap_size  = kwargs['heatmap_size']
        self.image_height  = self.final_shape[0]
        self.image_width   = self.final_shape[1]
        self.stride        = (self.image_width/self.heatmap_size[0],
                            self.image_height/self.heatmap_size[1])#effective stride of the entire network
        self.num_keypoints = 21 #21 annotated hand keypoints

        self.sc = kwargs.get('sc', 2.2)
        self.joint_names = ['Wrist', 'Thumb_k', 'Thumb_b', 'Thumb_m', 'Thumb_t', \
                               'Index_k', 'Index_b', 'Index_m', 'Index_t', \
                               'Middle_k', 'Middle_b', 'Middle_m', 'Middle_t', \
                               'Ring_k', 'Ring_b', 'Ring_m', 'Ring_t', \
                               'Pinky_k', 'Pinky_b', 'Pinky_m', 'Pinky_t']

        self.neighbor_link = [[1,1], [1,2], [2,3], [3,4],
                              [0,5], [5,6], [6,7], [7,8],
                              [0,9], [9,10], [10,11], [11,12],
                              [0,13], [13,14],[14,15], [15,16],
                              [0,17], [17,18], [18,19], [19,20]]
        #Colors RGB
        self.colors        = [[187,38,26],[187,38,26],[187,38,26],[187,38,26],
                              [172,201,63],[172,201,63],[172,201,63],[172,201,63],
                              [92,200,97],[92,200,97],[92,200,97],[92,200,97],
                              [28,84,197],[28,84,197],[28,84,197],[28,84,197],
                              [149,40,197],[149,40,197],[149,40,197],[149,40,197]]

        self.viz = kwargs['viz']

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        print('{} samples in {}'.format(len(self.samples), self.load_type))

        self.new_samples = []
        for idx, item in enumerate(self.samples):
            width, height = item['frame_size']
            for frm in item['frames']:
                for obj in frm['objs']:
                    kpts  = np.array(obj['hand_pts']).reshape(self.num_keypoints,3)

                    if np.any(np.array(obj['bbox']) < 0):
                        #A keypoint is occluded if either the x or y coordinate is less than 0
                        occ_x = kpts[:,0] < 0
                        occ_y = kpts[:,1] < 0
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_x, np.logical_or(occ_y, occ_c))
                        
                        obj['occ'] = occ

                    elif np.any(kpts[:,0] > width):
                        #A keypoint is occluded if the x coordinate is greater than image width
                        occ_x = kpts[:,0] > width 
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_x, occ_c)

                        obj['occ'] = occ

                    elif np.any(kpts[:,1] > height):
                        #A keypoint is occluded if the y coordinate is greater than image height
                        occ_y = kpts[:,1] > height 
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_y, occ_c)

                        obj['occ'] = occ
                    
                    #Don't keep samples with less than 2 keypoints visible
                    if sum(obj['occ']) >= (self.num_keypoints - 1)-1:
                        continue 

                    self.new_samples.append(item) 
        
        self.samples = self.new_samples
        del self.new_samples

        print('{} filtered samples in {}'.format(len(self.samples), self.load_type))

    #Adapted from: https://github.com/microsoft/human-pose-estimation.pytorch
    def generate_target(self, joints):
        ''' 
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, -1] 

        target = np.zeros((self.num_keypoints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3 

        for joint_id in range(self.num_keypoints):
            mu_x = int(joints[joint_id][0] / self.stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / self.stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] 
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
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
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)) 

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data      = []
        vid_data        = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data       = np.zeros((self.clip_length, 4))-1
        hand_crops      = np.zeros((self.clip_length, 4))-1
        hand_pts_coords = np.zeros((self.clip_length, self.num_keypoints, 2))-1
        org_hand_pts    = np.zeros((self.clip_length, self.num_keypoints, 2))-1
        labels          = np.zeros((self.clip_length))-1
        occlusions      = np.zeros((self.clip_length, self.num_keypoints), dtype=np.int32)-1 #21 keypoints
        padding         = np.zeros((self.clip_length, 4), dtype=np.int32) #pl, pt, pr, pb

        target          = np.zeros((self.clip_length, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)-1
        target_weight   = np.zeros((self.clip_length, self.num_keypoints, 1), dtype=np.float32)-1

        for frame_ind in range(len(vid_info['frames'])):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            #print('frame_path: {}'.format(frame_path))
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                trackid   = 0 #no trackids, image dataset 
                label     = 0 if obj['c'] == 'left' else 1 #0: left hand, 1: right hand
                occluded  = obj['occ']
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                hand_pts  = obj['hand_pts'] #21 points (x,y,valid)
                head_size = obj['head_size'] #max dim of tighest box around head

                #During training square patch is sc*B where B is max(obj_bbox)
                if self.load_type == 'train':
                    B = max(obj_bbox[2]-obj_bbox[0], obj_bbox[3]-obj_bbox[1])
                    #hand_size = np.random.uniform(1.5, 2.5, size=1) * B
                    hand_size = self.sc*B
                else: #During testing B is 0.7*head_size
                    B = 0.7*head_size
                    hand_size = self.sc*B

                hand_ctr = [obj_bbox[0]/2+obj_bbox[2]/2, obj_bbox[1]/2+obj_bbox[3]/2]
                #hand_size = sc*B
                xtl       = int(hand_ctr[0]-hand_size/2)
                ytl       = int(hand_ctr[1]-hand_size/2)
                xbr       = int(hand_ctr[0]+hand_size/2)
                ybr       = int(hand_ctr[1]+hand_size/2)

                #Pad images so hand is still in center of crop
                pl = pt = pr = pb = 0
                if xtl < 0:
                    pl = abs(xtl)
                if ytl < 0:
                    pt = abs(ytl)
                if xbr > width:
                    pr = abs(width - xbr)
                if ybr > height:
                    pb = abs(height - ybr)

                hand_crop   = [xtl+pl, ytl+pt, xbr, ybr]

                #incase annotations include invalid coords
                hand_pts = np.array(hand_pts, dtype=np.int32)[:,:2] #also convert to int, inconsistencies when using floating points
                #hand_pts[:,0] = np.clip(hand_pts[:,0], 0, width)
                #hand_pts[:,1] = np.clip(hand_pts[:,1], 0, height)

                bbox_data[frame_ind]        = obj_bbox
                labels[frame_ind]           = label 
                hand_pts_coords[frame_ind]  = hand_pts
                hand_pts_coords[frame_ind,:,0] += pl
                hand_pts_coords[frame_ind,:,1] += pt
                hand_crops[frame_ind]       = hand_crop
                org_hand_pts[frame_ind]     = hand_pts
                occlusions[frame_ind]       = occluded
                padding[frame_ind]          = [pl, pt, pr, pb]

            # Load frame, convert to RGB from BGR
            full_image = cv2.imread(os.path.join(base_path, frame_path))[...,::-1]
            input_data.append(full_image)
            #input_data.append(cv2.imread(os.path.join(base_path, frame_path))) #Keep as BGR

        #Crop hand and resize, perform same transforms to ground truth keypoints
        mask = [True if(1-o) else False for o in occluded] #need a mask because invalid keypoints messes up the preprocessing 
        vid_data, temp, _ = self.transforms(cv2.copyMakeBorder(input_data[frame_ind], pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)[None], {'bbox_data':hand_pts_coords[None, :, mask], 'hand_crop':hand_crop, 'label':labels})
        hand_pts_coords[frame_ind, mask] = temp
        hand_pts = np.concatenate((hand_pts_coords, np.array(mask)[None,:,None]), axis=-1)

        obj_trgt, obj_trgt_wght = self.generate_target(hand_pts[frame_ind])
        target[frame_ind] = obj_trgt
        target_weight[frame_ind] = obj_trgt_wght 

        target = torch.tensor(target)
        target_weight = torch.tensor(target_weight)

        #Add augmentation for prior, since it's the same as current crop
        #Gaussian noise added to each keypoint position
        noise_mu = 0
        noise_std = 8
        noise = np.zeros_like(hand_pts)
        noise[frame_ind,:,:2] = noise_std*np.random.randn(self.num_keypoints,2) + noise_mu
        #Zero out prior keypoint w/ probability p
        p = 0.20
        keep_prob = (np.random.rand(self.num_keypoints,1) > p)
        jitter_hand_pts = (hand_pts + noise) * keep_prob
        
        aux_input, _ = self.generate_target(jitter_hand_pts[frame_ind])
        aux_input = torch.from_numpy(aux_input).unsqueeze(0) 
        aux_data  = vid_data
        '''
        import matplotlib.pyplot as plt

        extent = np.int(0), np.int(self.heatmap_size[0]), np.int(0), np.int(self.heatmap_size[1])
        plt.subplot(1,3,1)
        vis = vid_data[0].numpy()
        mean = np.array([[[123.675,116.28,103.52]]])
        std = np.array([[[58.395,57.12,57.375]]])
        vis = np.clip(((vis*std)+mean)/255,0,1)
        plt.imshow(vis)
        plt.scatter(hand_pts_coords[frame_ind,:,0], hand_pts_coords[frame_ind,:,1])

        plt.subplot(1,3,2)
        plt.imshow(vis, interpolation='none', extent=extent)
        plt.imshow(torch.max(target[0], dim=0)[0].numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)

        plt.subplot(1,3,3)
        plt.imshow(vis, interpolation='none', extent=extent)
        plt.imshow(torch.max(aux_input[0], dim=0)[0].numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
        plt.show()
        '''

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)
        aux_data = aux_data.permute(3, 0, 1, 2)
        #vid_data = vid_data.squeeze(1) #Remove frame dimension, b/c this is an image dataset

        #Params needed for appropriate 'transforms', even though it's the same crop
        params = {'input_crop':hand_crops,
                  'padding':padding,
                  'prior_crop':hand_crops[0].astype(np.int),
                  'prior_pad':padding[0].astype(np.int),
                  'load_type': self.load_type, 
                  'frame_size':np.array(vid_size),
                  }
        ret_dict = dict() 
        ret_dict['data']       = [vid_data, aux_input, aux_data, params]

        annot_dict = dict()
        if self.viz:
            annot_dict['data']    = np.copy(full_image) #for visualization purposes only
        annot_dict['bbox']        = bbox_data
        annot_dict['input_crop']  = hand_crops 
        annot_dict['key_pts']     = org_hand_pts 
        #annot_dict['temp']        = hand_pts_coords
        annot_dict['head_size']   = head_size 
        annot_dict['heatmaps']    = target
        annot_dict['heatmap_weights'] = target_weight
        annot_dict['padding']     = padding 
        annot_dict['labels']      = labels
        annot_dict['occ']         = occlusions
        annot_dict['frame_path']  = frame_path 
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['neighbor_link'] = torch.tensor(self.neighbor_link)
        annot_dict['link_colors']   = torch.tensor(self.colors)
        annot_dict['joint_names'] = self.joint_names 
        ret_dict['annots']     = annot_dict

        return ret_dict
