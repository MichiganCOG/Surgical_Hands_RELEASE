import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class Surgical_Hands(DetectionDataset):
    """
    Data annotated from publicly available surgical hand videos

    x training samples
    x testing samples
    """
    def __init__(self, *args, **kwargs):
        super(Surgical_Hands, self).__init__(*args, **kwargs)

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

        self.sc = kwargs['sc']
        self.mask_occ = False #Treat occluded keypoints as un-annotated, if False treat them as GT labels

        self.joint_names = ['wrist', 'thumb_k', 'thumb_b', 'thumb_m', 'thumb_t', \
                               'index_k', 'index_b', 'index_m', 'index_t', \
                               'middle_k', 'middle_b', 'middle_m', 'middle_t', \
                               'ring_k', 'ring_b', 'ring_m', 'ring_t', \
                               'pinky_k', 'pinky_b', 'pinky_m', 'pinky_t']

        self.neighbor_link = [[0,1], [1,2], [2,3], [3,4],
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

        self.categories = {'supercategory':'hand',
                           'id':2,
                           'name':'hand', #maybe distinguish between left/right hand?
                           'keypoints':self.joint_names,
                           'skeleton':torch.Tensor(self.neighbor_link)}

        self.viz = kwargs['viz']

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms
            #self.transforms = kwargs['model_obj'].test_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        print('{} samples in {}'.format(len(self.samples), self.load_type))

        self.new_samples = []
        for idx, item in enumerate(self.samples):
            width, height = item['frame_size']

            for frm in item['frames']:
                bbox_data = []

                if not frm['is_labeled']:
                    continue 

                for obj in frm['objs']:
                    kpts  = np.array(obj['hand_pts']).reshape(self.num_keypoints,3)
                    #kpts - (x,y,visibility)
                    #visibility: 0 - unannotated, 1 - occluded, 2 - visible

                    if np.any(np.array(obj['bbox']) < 0):
                        #A keypoint is occluded if either the x or y coordinate is less than 0
                        occ_x = kpts[:,0] < 0
                        occ_y = kpts[:,1] < 0
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_x, np.logical_or(occ_y, occ_c))
                        
                        obj['occ'] = occ

                    elif np.any(kpts[:,0] > width):
                        #A keypoint is occluded if either the x coordinate is greater than image width
                        occ_x = kpts[:,0] > width 
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_x, occ_c)

                        obj['occ'] = occ

                    elif np.any(kpts[:,1] > height):
                        #A keypoint is occluded if either the y coordinate is greater than image height
                        occ_y = kpts[:,1] > height 
                        occ_c = (kpts[:,2] == 0)
                        occ   = np.logical_or(occ_y, occ_c)

                        obj['occ'] = occ
                    
                    #Don't keep samples with less than 2 keypoints visible
                    if sum(obj['occ']) >= (self.num_keypoints - 1)-1:
                        continue 
                    
                    bbox_data.append(obj['bbox'])

                    new_item = {}
                    new_item['frames'] = [{'objs':[obj], 'img_path':frm['img_path'],\
                            'vid_id':frm['vid_id'], 'frame_id':frm['frame_id'], 'is_labeled':frm['is_labeled']}]
                    new_item['base_path']  = item['base_path']
                    new_item['frame_size'] = item['frame_size']

                    self.new_samples.append(new_item) 

                '''
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches 

                fig = plt.figure()
                ax = fig.add_subplot(111)

                base_path = item['base_path']
                frame_path = frm['img_path']
                vis = (cv2.imread(os.path.join(base_path, frame_path))[...,::-1])
                plt.imshow(vis)

                for bbox in bbox_data:
                    #tight bbox
                    xmin, ymin, xmax, ymax = bbox
                    rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)

                plt.show()
                '''

        self.samples = self.new_samples
        del self.new_samples

        print('{} filtered samples in {}'.format(len(self.samples), self.load_type))

        #Some collected statistics for each fold:
        num_left  = 0
        num_right = 0
        avg_bbox_area = 0
        avg_frame_area = 0
        avg_num_vis  = np.zeros(self.num_keypoints)
        avg_num_occ  = np.zeros(self.num_keypoints)
        avg_num_nvis = np.zeros(self.num_keypoints)
        for sample in self.samples:
            obj = sample['frames'][0]['objs'][0]

            if obj['c'] == 'right':
                num_right += 1
            else:
                num_left += 1

            avg_bbox_area += ((obj['bbox'][2]-obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1])/len(self.samples))

            avg_frame_area += ((sample['frame_size'][0] * sample['frame_size'][1])/len(self.samples))

            kpts = np.array(obj['hand_pts']).reshape(self.num_keypoints,3)

            avg_num_vis += ((kpts[:,2] == 2)/len(self.samples))
            avg_num_occ += ((kpts[:,2] == 1)/len(self.samples))
            avg_num_nvis += ((kpts[:,2] == 0)/len(self.samples))
        
        print('--'*30)
        print(self.json_path)
        print('Average bbox area: {:.2f} px, {:.2f}^2 px'.format(avg_bbox_area, np.sqrt(avg_bbox_area)))
        print('Average frame area: {:.2f} px, {:.2f}^2 px'.format(avg_frame_area, np.sqrt(avg_frame_area)))
        print('Percentage: {:.2f}'.format(avg_bbox_area/avg_frame_area))
        print('Average Percentage: Visible, Occluded, N/A')
        for idx, joint in enumerate(self.joint_names):
            print('{}: Vis: {:.2f}, Occ: {:.2f}, N/A: {:.2f}'.format(joint, avg_num_vis[idx], avg_num_occ[idx], avg_num_nvis[idx]))
        print('All joints: Vis: {:.2f}, Occ: {:.2f}, N/A: {:.2f}'.format(np.mean(avg_num_vis), np.mean(avg_num_occ), np.mean(avg_num_nvis)))
        print('--'*30)

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
        hand_pts_coords = np.zeros((self.clip_length, self.num_keypoints, 3))-1
        org_hand_pts    = np.zeros((self.clip_length, self.num_keypoints, 3))-1
        track_ids       = np.zeros((self.clip_length), dtype=np.int32)-1
        obj_ids         = np.zeros((self.clip_length), dtype=np.int64)-1
        labels          = np.zeros((self.clip_length))-1
        unannotated     = np.zeros((self.clip_length, 21), dtype=np.int32)-1 #21 keypoints
        visibility      = np.zeros((self.clip_length, 21), dtype=np.int32) #visibility: 0 - unannotated, 1 - occluded, 2 - visible
        padding         = np.zeros((self.clip_length, 4), dtype=np.int32) #pl, pt, pr, pb

        target          = np.zeros((self.clip_length, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)-1
        target_weight   = np.zeros((self.clip_length, self.num_keypoints, 1), dtype=np.float32)-1

        frame_ids   = np.zeros((self.clip_length), dtype=np.int64)
        frame_paths = []
        for frame_ind in range(len(vid_info['frames'])):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            vid_id         = frame['vid_id']
            frame_id       = int(frame['frame_id'])
            #print('frame_path: {}'.format(frame_path))
            
            # Extract bbox and label data from video info
            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id

            # Load frame, convert to RGB from BGR and normalize from 0 to 1
            input_data = cv2.imread(frame_path)[...,::-1]

            for obj in frame['objs']:
                trackid   = obj['trackid'] #Let's ignore trackid for now, only one annotation per image
                obj_id    = obj['id']
                label     = 0 if obj['c'] == 'left' else 1 #0: left hand, 1: right hand
                unann     = obj['occ']
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                hand_pts  = obj['hand_pts'] #21 points (x,y,visibility)

                xmin, ymin, xmax, ymax = obj_bbox

                #ensure bounding box encompasses all keypoints - error occurs otherwise
                hand_pts = np.array(hand_pts).reshape((self.num_keypoints, 3)) 
                _mask = hand_pts[:,-1] > 0 

                xpt_max, ypt_max, _ = np.max(hand_pts[_mask], axis=0)
                xpt_min, ypt_min, _ = np.min(hand_pts[_mask], axis=0)

                xtl_adjust = np.clip(xmin-xpt_min, a_min=0, a_max=None)
                ytl_adjust = np.clip(ymin-ypt_min, a_min=0, a_max=None)
                xbr_adjust = np.clip(xpt_max-xmax, a_min=0, a_max=None)
                ybr_adjust = np.clip(ypt_max-ymax, a_min=0, a_max=None)

                xmin -= xtl_adjust
                ymin -= ytl_adjust
                xmax += xbr_adjust
                ymax += ybr_adjust

                #expand area around bbox
                sc = self.sc  
                w = xmax - xmin
                h = ymax - ymin 
                cx = xmin + w/2 
                cy = ymin + h/2 

                w *= sc  
                h *= sc  

                xmin = int(cx - (w/2))
                ymin = int(cy - (h/2))
                xmax = int(cx + (w/2))
                ymax = int(cy + (h/2))

                #Pad images so hand is still in center of crop
                pl = pt = pr = pb = 0 
                if xmin < 0:
                    pl = abs(xmin)
                if ymin < 0:
                    pt = abs(ymin)
                if xmax > (width + pl):
                    pr = abs(width - xmax) 
                if ymax > (height + pt):
                    pb = abs(height - ymax)

                hand_crop = [xmin+pl, ymin+pt, xmax, ymax]
                
                #incase annotations include invalid coords
                visibility = hand_pts[:,-1]

                if self.mask_occ:
                    for i, v in enumerate(visibility):
                        if v == 1:
                            unann[i] = True 

                org_hand_pts[frame_ind]   = hand_pts
                hand_pts += np.array([[pl,pt,0]]) #Adjust keypoints by padding

                #hand_pts[:,0] = np.clip(hand_pts[:,0], 0, width)
                #hand_pts[:,1] = np.clip(hand_pts[:,1], 0, height)
                hand_pts[:,2] = np.clip(hand_pts[:,2], 0, 1)

                #Let's make the obj_id numeric only
                obj_id = int(''.join((obj_id.split('_')[-4:])))

                bbox_data[frame_ind]     = obj_bbox
                track_ids[frame_ind]     = trackid 
                obj_ids[frame_ind]       = obj_id 
                labels[frame_ind]        = label 
                hand_pts_coords[frame_ind] = hand_pts 
                hand_crops[frame_ind]     = hand_crop
                unannotated[frame_ind]    = unann
                padding[frame_ind]        = [pl, pt, pr, pb]

        #Crop hand and resize, perform same transforms to ground truth keypoints
        mask = [True if(1-o) else False for o in unann] #need a mask because invalid keypoints messes up the preprocessing 

        vid_data, temp, _ = self.transforms(cv2.copyMakeBorder(input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)[None], {'bbox_data':hand_pts_coords[None,:,mask,:2], 'hand_crop':hand_crop, 'label':labels})
        hand_pts_coords[None,:,mask,:2] = temp

        obj_trgt, obj_trgt_wght = self.generate_target(hand_pts_coords[0])
        target[frame_ind] = obj_trgt
        target_weight[frame_ind] = obj_trgt_wght 

        target = torch.tensor(target)
        target_weight = torch.tensor(target_weight)

        '''
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches 

        fig = plt.figure()
        ax1 = fig.add_subplot(131)

        pl, pt, pr, pb = padding[frame_ind]
        ax1.imshow(cv2.copyMakeBorder(input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0))

        xmin, ymin, xmax, ymax = bbox_data[frame_ind]
        rect1 = patches.Rectangle((xmin+pl, ymin+pt), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
        xmin, ymin, xmax, ymax = hand_crops[frame_ind]
        rect2 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax1.add_patch(rect2)

        extent = np.int(0), np.int(self.heatmap_size[0]), np.int(0), np.int(self.heatmap_size[1])
        ax2 = plt.subplot(132)
        vis = vid_data[0].numpy()
        mean = np.array([[[123.675,116.28,103.52]]])
        std = np.array([[[58.395,57.12,57.375]]])
        vis = np.clip(((vis*std)+mean)/255,0,1)
        ax2.imshow(vis, interpolation='none', extent=extent)

        ax3 = plt.subplot(133)
        ax3.imshow(vis, interpolation='none', extent=extent)
        ax3.imshow(torch.max(target[0], dim=0)[0].numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)

        plt.show()
        '''

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)
        #vid_data = vid_data.squeeze(1) #Remove frame dimension, b/c this is an image dataset

        ret_dict = dict() 
        ret_dict['data']       = vid_data 
        annot_dict = dict()
        #annot_dict['data']        = cv2.cvtColor(np.copy(input_data), cv2.COLOR_BGR2RGB) #TODO:for visualization purposes only
        if self.viz:
            annot_dict['data']        = cv2.cvtColor(np.copy(input_data), cv2.COLOR_BGR2RGB) #TODO:for visualization purposes only
            #annot_dict['data']   = vid_data 
        annot_dict['key_pts']     = org_hand_pts
        #annot_dict['temp']        = hand_pts_coords 
        annot_dict['heatmaps']    = target
        annot_dict['heatmap_weights'] = target_weight
        annot_dict['bbox']        = bbox_data
        annot_dict['input_crop']   = hand_crops
        annot_dict['padding']     = padding
        annot_dict['track_ids']   = track_ids
        annot_dict['obj_ids']     = obj_ids 
        annot_dict['labels']      = labels
        annot_dict['occ']         = unannotated
        annot_dict['frame_path']  = frame_paths
        annot_dict['frame_ids']   = frame_ids 
        annot_dict['nframes']     = 1 #not useful
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['raw_frame_size'] = np.array(vid_size)
        annot_dict['load_type']   = self.load_type 
        annot_dict['neighbor_link'] = torch.tensor(self.neighbor_link)
        annot_dict['link_colors']   = torch.tensor(self.colors)
        annot_dict['vid_id']      = frame['vid_id']
        annot_dict['joint_names'] = self.joint_names 
        annot_dict['categories'] = self.categories
        ret_dict['annots']     = annot_dict

        return ret_dict
