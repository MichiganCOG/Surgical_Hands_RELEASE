#Output GT heatmap as an auxiliary input
#Single object only
#Additionally output the previous image crop
#at the same location (but centered on the (current?) object)
import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json
import math

class Surgical_Hands_v2(DetectionDataset):
    """
    Data annotated from publicly available surgical hand videos

    x training samples
    x testing samples
    """
    def __init__(self, *args, **kwargs):
        super(Surgical_Hands_v2, self).__init__(*args, **kwargs)

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

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        #Track statistics of hand positions through dataset
        avg_hand_pts = np.zeros((self.num_keypoints, 2))
        num_hand_pts = np.zeros((self.num_keypoints,1))

        print('{} samples in {}'.format(len(self.samples), self.load_type))
        self.new_samples = []
        max_clip_length = 0 

        self.img_id_to_kpts = {} #Mapping between images and keypoints within them
        self.t1_to_t0 = {} #Point to the previous image. First image points to itself
        min_temporal_dist = kwargs.get('min_temporal_distance', 4) #accidentally used multiple names 
        min_temporal_dist = kwargs.get('min_temporal_dist', 4) #final name

        vid_id_to_frames = {} #all the labeled frames in each vid_id
        vid_id_to_path   = {}  
        prev_vid_id   = None
        prev_frame_id = None
        for idx, item in enumerate(self.samples):
            width, height = item['frame_size']

            vid_id = item['frames'][0]['vid_id']

            labeled_frames  = vid_id_to_frames.get(vid_id, [])
            lbl_frame_paths = vid_id_to_path.get(vid_id, [])
            for frm in item['frames']:
                bbox_data = []

                if not frm['is_labeled']:
                    continue 

                frame_id  = int(frm['frame_id'])
                frame_pth = frm['img_path']
                labeled_frames.append(frame_id)
                lbl_frame_paths.append(frame_pth)

                if frame_id not in self.img_id_to_kpts:
                    self.img_id_to_kpts[frame_id] = {}

                for obj in frm['objs']:
                    kpts  = np.array(obj['hand_pts']).reshape(self.num_keypoints,3)
                    #kpts - (x,y,visibility)
                    #visibility: 0 - unannotated, 1 - occluded, 2 - visible

                    if prev_vid_id != vid_id:
                        #self.t1_to_t0[frame_id] = {'frame_path':frame_pth, 'frame_id':frame_id} #first frame points to itself
                        self.t1_to_t0[frame_id] = None #first frame points to None
                    elif frame_id != prev_frame_id:
                        d = abs(frame_id - np.array(labeled_frames))
                        valid = np.array(labeled_frames)[d >= min_temporal_dist] #frames atleast t frames away

                        #selected_frame_id = max(valid, default=frame_id) #point to the closest valid frame, defaults to itself
                        selected_frame_id = max(valid, default=None)

                        try:
                            idx = labeled_frames.index(selected_frame_id)
                            selected_frame_path = lbl_frame_paths[idx]
                        except ValueError: #selected_frame_id is None, i.e. needs to be atleast t frames
                            selected_frame_path = None
                        self.t1_to_t0[frame_id] = {'frame_path':selected_frame_path, \
                                'frame_id':selected_frame_id} #point to the closest valid frame, defaults to None 

                    #########Generate keypoints for aux input
                    trackid   = obj['trackid']
                    unann     = obj['occ']
                    obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                    hand_pts  = obj['hand_pts'] #21 points (x,y,visibility)

                    if obj_bbox == []:
                        obj_bbox = np.zeros(4)

                    xmin, ymin, xmax, ymax = obj_bbox

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
                    hand_pts = np.array(hand_pts).reshape((self.num_keypoints, 3))
                    visibility = hand_pts[:,-1]

                    if self.mask_occ:
                        for i, v in enumerate(visibility):
                            if v == 1:
                                unann[i] = True 

                    hand_pts += np.array([[pl,pt,0]]) #Adjust keypoints by padding
                    
                    #Crop hand and resize, perform same transforms to ground truth keypoints
                    mask = [True if(1-o) else False for o in unann] #need a mask because invalid keypoints messes up the preprocessing 

                    self.img_id_to_kpts[frame_id][trackid] = {'hand_pts':hand_pts, 'bbox':obj['bbox'], 'center': [cx,cy],\
                            'mask':mask, 'crop':hand_crop, 'padding':[pl,pt,pr,pb]}
                    

                    #Track keypoint statistics, based on expected padding and crop
                    vis = (visibility[:,None] > 0)
                    avg_hand_pts += (hand_pts[:,:2]/np.array([[w,h]]) * vis)
                    num_hand_pts += vis 
                    ######################

                    prev_vid_id   = vid_id
                    prev_frame_id = frame_id 

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
                            'vid_id':vid_id, 'frame_id':frame_id, 'is_labeled':frm['is_labeled']}]
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
                vis = (cv2.imread(frame_path)[...,::-1])
                plt.imshow(vis)

                for bbox in bbox_data:
                    #tight bbox
                    xmin, ymin, xmax, ymax = bbox
                    rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)

                plt.show()
                '''
            
            vid_id_to_frames[vid_id] = labeled_frames
            vid_id_to_path[vid_id]   = lbl_frame_paths

        self.samples = self.new_samples
        del self.new_samples

        print('{} filtered samples in {}'.format(len(self.samples), self.load_type))

        '''
        #Calculate displacement of hands between min_temporal_dist frames
        obj_dists = []
        for t1 in self.t1_to_t0.keys():
            t0 = self.t1_to_t0[t1]

            if t0 is None or t0['frame_id'] is None:
                continue

            objs_t1 = self.img_id_to_kpts[t1]
            objs_t0 = self.img_id_to_kpts[t0['frame_id']]

            for tid, obj_t1 in objs_t1.items():
                obj_t0 = objs_t0.get(tid, None)

                if obj_t0 is None:
                    continue

                kpt_t1_mean = np.mean(obj_t1['hand_pts'][obj_t1['mask']], axis=0)[:2]
                kpt_t0_mean = np.mean(obj_t0['hand_pts'][obj_t0['mask']], axis=0)[:2]
                dist = np.linalg.norm(kpt_t0_mean - kpt_t1_mean) #Euclidean distance between both centers

                if math.isnan(dist):
                    continue 

                obj_dists.append(dist)
        
        print('Mean dist: {}'.format(np.mean(obj_dists)))
        print('Median dist: {}'.format(np.median(obj_dists)))
        print('Max dist: {}'.format(np.max(obj_dists)))
        import pdb; pdb.set_trace()
        
        import matplotlib.pyplot as plt

        plt.hist(obj_dists, bins=30)
        plt.show()
        '''

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
        org_hand_pts    = np.zeros((self.clip_length, self.num_keypoints, 2))-1
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
            frame_id       = frame['frame_id']
            
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
                vis = hand_pts[:,-1]

                if self.mask_occ:
                    for i, v in enumerate(vis):
                        if v == 1:
                            unann[i] = True 

                org_hand_pts[frame_ind]   = hand_pts[:,:2]
                hand_pts += np.array([[pl,pt,0]]) #Adjust keypoints by padding

                #hand_pts[:,0] = np.clip(hand_pts[:,0], 0, width)
                #hand_pts[:,1] = np.clip(hand_pts[:,1], 0, height)
                hand_pts[:,2] = np.clip(hand_pts[:,2], 0, 1)

                #Let's make the obj_id numeric only
                obj_id = int(''.join((obj_id.split('_')[-4:])))

                bbox_data[frame_ind]     = obj_bbox
                obj_ids[frame_ind]       = obj_id
                labels[frame_ind]           = label 
                hand_pts_coords[frame_ind] = hand_pts 
                hand_crops[frame_ind]     = hand_crop
                unannotated[frame_ind]    = unann
                padding[frame_ind]        = [pl, pt, pr, pb]

        #Crop hand and resize, perform same transforms to ground truth keypoints
        mask = [True if(1-o) else False for o in unann] #need a mask because invalid keypoints messes up the preprocessing 

        vid_data, temp, out_params = self.transforms(cv2.copyMakeBorder(input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)[None], {'bbox_data':hand_pts_coords[None,:,mask,:2], 'hand_crop':hand_crop, 'label':labels})

        flipped = out_params['flip']
        angle   = out_params.get('out_rot', None)
        hand_pts_coords[None,:,mask,:2] = temp 

        obj_trgt, obj_trgt_wght = self.generate_target(hand_pts_coords[0])
        target[frame_ind] = obj_trgt
        target_weight[frame_ind] = obj_trgt_wght 

        aux_input_data  = np.zeros((height, width, 3), dtype=np.float32)
        aux_input = np.zeros((self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        aux_data  = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3), dtype=np.float32)
        aux_pts_coords = np.zeros((self.clip_length, self.num_keypoints, 3))-1
        obj_crop = obj_pad = [-1,-1,-1,-1] 
        if not self.t1_to_t0[frame_id] is None and not self.t1_to_t0[frame_id]['frame_id'] is None: #1st frame may not have a prior
            #Extract and resize this object's keypoints on given frame
            img_objs = self.img_id_to_kpts[self.t1_to_t0[frame_id]['frame_id']]
            for key, obj in img_objs.items():
                if key != trackid: #Only this current object
                    continue 

                obj_kpts   = np.copy(obj['hand_pts'])

                aux_frame_path = self.t1_to_t0[frame_id]['frame_path']
                #print('Aux frame path: {}'.format(aux_frame_path))

                '''
                obj_kpts[:,2] = np.clip(obj_kpts[:,2], 0, 1)
                #Apply same transformation to keypoints TODO 

                #flip prior if target if flip augmentation was applied
                if flipped:
                    #flip all x-positions
                    obj_kpts[:,0] = self.image_width - obj_kpts[:,0]

                obj_trgt, _ = self.generate_target(obj_kpts)
                aux_input = obj_trgt
                '''

            try:
                obj_kpts   = np.copy(img_objs[trackid]['hand_pts'])
                obj_mask   = img_objs[trackid]['mask']
                obj_bbox   = img_objs[trackid]['bbox']
                obj_center = img_objs[trackid]['center']
                obj_crop   = img_objs[trackid]['crop']
                pl,pt,pr,pb = img_objs[trackid]['padding']

                obj_pad = [pl, pt, pr, pb]

                obj_kpts[:,2] = np.clip(obj_kpts[:,2], 0, 1)
                aux_pts_coords[0] = obj_kpts

                aux_input_data = cv2.imread(os.path.join(base_path, aux_frame_path))[...,::-1]
                aux_data, temp, out_params = self.transforms(cv2.copyMakeBorder(aux_input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)[None], {'bbox_data':obj_kpts[None,None,obj_mask,:2], 'hand_crop':obj_crop, 'label':labels, 'in_rot':angle})

                aux_data = np.array(aux_data)
                #this section may be unnecessary
                '''
                aux_pts_coords[None,:,obj_mask,:2] = temp 

                #flip prior, if target flip augmentation was applied
                if flipped and not out_params['flip']:
                    aux_data[0] = cv2.flip(aux_data[0], 1)

                    #flip all x-positions
                    aux_pts_coords[...,0] = self.image_width - aux_pts_coords[...,0]
                elif not flipped and out_params['flip']:
                    #Not flipped in target, but flipped in prior
                    #ideally remove this randomization

                    aux_data[0] = cv2.flip(aux_data[0], 1)

                    #flip all x-positions
                    aux_pts_coords[...,0] = self.image_width - aux_pts_coords[...,0]

                #transform keypoints to fit current image crop, and then generate that as a heatmap prior
                #important for small variations in bounding box placement and aspect ratio

                ###Unscale from time t-1 params### 
                if flipped:
                    aux_pts_coords[...,0] = (self.image_width - aux_pts_coords[...,0]) #undo any flipping

                #scale coordinates to crop size
                obj_crop_h = (obj_crop[3]-obj_crop[1])
                obj_crop_w = (obj_crop[2]-obj_crop[0])
                aux_pts_coords[:,:,0] *= (obj_crop_w/self.image_width)
                aux_pts_coords[:,:,1] *= (obj_crop_h/self.image_height)

                #approx to int
                aux_pts_coords = np.ceil(aux_pts_coords)

                #Undo crop
                aux_pts_coords[:,:,0] += obj_crop[0]
                aux_pts_coords[:,:,1] += obj_crop[1]

                '''
                #Subtract padding if was added
                aux_pts_coords = np.copy(obj_kpts)[None]
                aux_pts_coords[:,:,0] -= pl 
                aux_pts_coords[:,:,1] -= pt

                ###Rescale to time t properties###
                crop_xmin, crop_ymin, crop_xmax, crop_ymax = hand_crop 
                pl, pt, pb, pr = padding[frame_ind]

                mask = np.array(aux_pts_coords[...,-1], dtype=np.bool)

                #adjust keypoints by crop
                aux_pts_coords[mask,0], aux_pts_coords[mask,1] = crop_coords(aux_pts_coords[mask,0]+pl, aux_pts_coords[mask,1]+pt, crop_xmin, crop_ymin, crop_xmax, crop_ymax)

                #add rotation if necessary
                if angle is not None:
                    aux_pts_coords[:,mask.squeeze(),:2] = rotate_coords(aux_pts_coords[:,mask.squeeze(),:2],\
                            (crop_ymax-crop_ymin, crop_xmax-crop_xmin), angle)

                #adjust for resized input image
                aux_pts_coords[mask,0], aux_pts_coords[mask,1] = resize_pt_coords(aux_pts_coords[mask,0], aux_pts_coords[mask,1], (crop_ymax-crop_ymin, crop_xmax-crop_xmin), (self.image_height,self.image_width))

                if flipped:
                    if not out_params['flip']:
                        aux_data[0] = cv2.flip(aux_data[0], 1)

                    aux_pts_coords[...,0] = (self.image_width - aux_pts_coords[...,0]) #Add flipping, if any
                    temp[...,0] = (self.image_width - temp[...,0])
                elif not flipped and out_params['flip']:
                    #Not flipped in target, but flipped in prior
                    #ideally remove this randomization

                    aux_data[0] = cv2.flip(aux_data[0], 1)
                
                obj_trgt, _ = self.generate_target(aux_pts_coords[0])
                aux_input = obj_trgt
            except KeyError: #No kpts to crop around image or object doesn't exist at frame
                pass

        '''
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches 

        fig = plt.figure(1)
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(334)

        pl, pt, pr, pb = padding[frame_ind]
        ax1.imshow(cv2.copyMakeBorder(input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0))
        #ax1.imshow(input_data)

        xmin, ymin, xmax, ymax = bbox_data[frame_ind]
        rect1 = patches.Rectangle((xmin+pl, ymin+pt), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
        xmin, ymin, xmax, ymax = hand_crops[frame_ind]
        rect2 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax1.add_patch(rect2)

        extent = np.int(0), np.int(self.heatmap_size[0]), np.int(0), np.int(self.heatmap_size[1])

        vis = vid_data[0].numpy()
        mean = np.array([[[123.675,116.28,103.52]]])
        std = np.array([[[58.395,57.12,57.375]]])
        vis = np.clip(((vis*std)+mean)/255,0,1)
        ax2.imshow(vis, interpolation='none', extent=extent)
        title = 'Flipped' if flipped else ''
        ax2.title.set_text(title)
        ax2.scatter(48,48)

        ax3 = fig.add_subplot(332)
        aux_vis = np.clip(((aux_data[0]*std)+mean)/255,0,1)
        ax3.imshow(aux_vis, interpolation='none', extent=extent)
        ax3.imshow(np.max(aux_input,axis=0), cmap='jet', alpha=0.5, interpolation='none', extent=extent) 
        x = (temp[0,0,:,0]*96/368).numpy()
        y = (temp[0,0,:,1]*96/368).numpy()
        #ax3.scatter(x,y,c='w')
        #ax3.title.set_text('frame {}'.format(self.t1_to_t0[frame_id]['frame_id']))
        #ax3.colorbar()

        ax4 = fig.add_subplot(335)
        ax4.imshow(vis, interpolation='none', extent=extent)
        ax4.imshow(np.max(target[0],axis=0), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
        ax4.title.set_text('frame {}'.format(frame_id))
        ax4.scatter(48,48)
        #ax4.colorbar()

        ax5 = fig.add_subplot(333)
        if obj_crop != [-1,-1,-1,-1]:
            pl, pt, pr, pb = obj_pad
            ax5.imshow(cv2.copyMakeBorder(aux_input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0))
            xmin, ymin, xmax, ymax = obj_bbox
            rect1 = patches.Rectangle((xmin+pl, ymin+pt), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
            xmin, ymin, xmax, ymax = obj_crop
            rect2 = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax5.add_patch(rect1)
            ax5.add_patch(rect2)

        ax6 = fig.add_subplot(336)
        ax6.imshow(aux_vis, interpolation='none', extent=extent)
        title = 'Flipped' if flipped else ''
        ax6.title.set_text(title)
        x = temp[0,0,:,0].numpy()
        y = temp[0,0,:,1].numpy()
        ax6.scatter(48,48)

        ax7 = fig.add_subplot(338)
        ax7.imshow(vis, interpolation='none', extent=extent)
        ax7.imshow(np.max(aux_input,axis=0), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
        ax7.title.set_text('previous anno projected onto current frame')

        plt.show()
        '''

        target        = torch.tensor(target)
        target_weight = torch.tensor(target_weight)

        #aux input will be the heatmap of the same object at designated (previous) frame
        aux_input = torch.from_numpy(aux_input).unsqueeze(0)
        aux_data = torch.from_numpy(aux_data)

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)
        aux_data = aux_data.permute(3, 0, 1, 2)

        #Some params for frame t, needed for appropriate transforms in forward function
        params = {'input_crop':hand_crops,
                  'padding':padding,
                  'prior_crop':np.array(obj_crop),
                  'prior_pad':np.array(obj_pad),
                  'load_type': self.load_type, 
                  'frame_size':np.array(vid_size),
                  }

        ret_dict = dict() 
        ret_dict['data']       = [vid_data, aux_input, aux_data, params]
        annot_dict = dict()
        if self.viz:
            #annot_dict['data']  = cv2.cvtColor(np.copy(input_data), cv2.COLOR_BGR2RGB) #TODO:for visualization purposes only
            annot_dict['data']   = vid_data 
        annot_dict['key_pts']     = org_hand_pts
        #annot_dict['temp']        = hand_pts_coords 
        annot_dict['heatmaps']    = target
        annot_dict['heatmap_weights'] = target_weight
        annot_dict['bbox']        = bbox_data
        annot_dict['input_crop']   = hand_crops
        annot_dict['padding']     = padding
        annot_dict['obj_ids']     = obj_ids
        annot_dict['labels']      = labels
        annot_dict['occ']         = unannotated
        annot_dict['frame_path']  = frame_paths
        annot_dict['frame_ids']   = frame_ids 
        annot_dict['nframes']     = 1 #not useful
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['neighbor_link'] = torch.tensor(self.neighbor_link)
        annot_dict['link_colors']   = torch.tensor(self.colors)
        annot_dict['vid_id']      = frame['vid_id']
        annot_dict['joint_names'] = self.joint_names 
        annot_dict['categories'] = self.categories
        ret_dict['annots']     = annot_dict

        return ret_dict

def crop_coords(x, y, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        #TODO: Not sure if this is avoidable. Prev points may not be completely inside curr bounding box
        #if np.any(x > crop_xmax) or np.any(x < crop_xmin) or np.any(y > crop_ymax) or np.any(y < crop_ymin):
        #    return -1*np.ones(x.shape), -1*np.ones(y.shape)

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

 ######
# Code from: https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def _cart2pol(point):
    x,y = point
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi) 

def _pol2cart(point):
    rho, phi = point
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
#####

def rotate_coords(bboxes, frame_shape, angle):
        angle = np.deg2rad(angle)
        bboxes_shape = bboxes.shape
        output_bboxes = np.zeros(bboxes_shape)-1
        frame_h, frame_w = frame_shape[0], frame_shape[1] 
        half_h = frame_h/2. 
        half_w = frame_w/2. 

        for bbox_ind in range(bboxes_shape[0]):
            x, y = bboxes[bbox_ind].transpose()

            pts  = (x-half_w, y-half_h)

            pts = _cart2pol(pts)

            pts = (pts[0], pts[1]-angle)

            pts = _pol2cart(pts)

            pts  = (pts[0]+half_w, pts[1]+half_h)

            output_bboxes[bbox_ind,:,0] = (np.clip(pts[0], 0, frame_w-1))
            output_bboxes[bbox_ind,:,1] = (np.clip(pts[1], 0, frame_h-1))

        return output_bboxes
