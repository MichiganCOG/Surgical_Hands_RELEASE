#Main difference between v1: This Dataloader outputs all objects on the current frame together
import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class PoseTrack18_v2(DetectionDataset):
    """
    PoseTrack 2018 Dataset 

    1,356 annotated sequences
    593 train
    170 validation
    375 test

    276,198 annotated poses in total

    Source: https://arxiv.org/abs/1710.10000

    0: Nose
    1: Head bottom
    2: Head top
    3: Left ear (not annotated - masked out)
    4: Right ear (not annotated - masked out) 
    5: Left shoulder
    6: Right shoulder
    7: Left elbow
    8: Right elbow
    9: Left wrist
    10: Right wrist
    11: Left hip
    12: Right hip
    13: Left knee
    14: Right knee
    15: Left ankle
    16: Right ankle 

    Left ear and right ears are not annotated in dataset
    """
    def __init__(self, *args, **kwargs):
        super(PoseTrack18_v2, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']
        self.json_path = kwargs['json_path']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        #Unique ids: 57, Max persons per frame: 36, Max persons per vid: 56
        self.max_objects   = 36
        self.sigma         = kwargs['gaussian_sigma']
        self.heatmap_size  = kwargs['heatmap_size']
        self.image_height = self.final_shape[0]
        self.image_width  = self.final_shape[1]
        self.stride        = (self.image_width/self.heatmap_size[0], \
                              self.image_height/self.heatmap_size[1]) #effective stride of the network
        
        self.sc = kwargs['sc'] #parameter search for the scaling. default = 1.25

        #self.num_keypoints = 17 + 1 #17 annotated body keypoints (+ added keypoint)
        self.num_keypoints = 17 #17 annotated body keypoints (but really 15)
        self.flip_pairs = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]

        self.sample_all_obj = kwargs['sample_all_obj']
        self.aspect_ratio = self.image_width / self.image_height
        self.pixel_std = 200

        self.joint_names = ['Nose', 'Head b', 'Head t', 'L Ear', 'R Ear', 'L Shoulder', 'R Shoulder',\
                            'L elbow', 'R elbow', 'L wrist', 'R wrist', 'L hip', 'R hip',\
                            'L knee', 'R knee', 'L ankle', 'R ankle']

        self.viz = kwargs['viz']
        
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        #Longest clip length is 150 (train) (labeled + unlabeled frames)
        #Longest (labeled) clip lengths are 31 (train), 60 (val), 98 (test)

        #Remove items without keypoint annotations
        #And separate every ground truth object into a separate sample. 
        #This reserves memory and allows stacking of inputs
        print('{} videos in {}'.format(len(self.samples), self.load_type))

        self.new_samples = []
        max_clip_length = 0

        for idx, item in enumerate(self.samples):
            segment_samples = {}

            frm_index = 0 #annotated frame index (needed later in metrics)
            for frm in item['frames']:
                
                if not frm['is_labeled']: #skip unlabeled frames
                    continue

                valid_objs = []

                for obj in frm['objs']:
                    if obj['occ']: #skip un-annotated persons
                        continue 

                    kpts = np.array(obj['keypoints']).reshape(self.num_keypoints, 3)
                    if np.sum(kpts[:,-1]) < 1: #skip if ALL keypoints are occluded or un-annotated, and if only 1 keypoint exists
                        continue 

                    valid_objs.append(obj)

                f = [{'objs':valid_objs, 'img_path':frm['img_path'], \
                        'nframes':frm['nframes'], 'frame_id':frm['frame_id'],\
                        'vid_id':frm['vid_id'], 'is_labeled':frm['is_labeled'],\
                        'categories':frm['categories'], 'frm_index':frm_index}]

                new_item = item.copy()
                new_item['frames'] = f

                segment_samples[frm['frame_id']] = new_item

                if len(new_item['frames']) > max_clip_length:
                    max_clip_length = len(new_item['frames'])

                frm_index += 1

            self.new_samples.extend(segment_samples.values())
        self.clip_length = max_clip_length 
        self.samples = self.new_samples
        del self.new_samples 

        print('Max clip length: {}'.format(max_clip_length))
        print('{} annotated objects in {}'.format(len(self.samples), self.load_type))

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
        
        base_path  = vid_info['base_path']
        vid_size   = vid_info['frame_size']
        vid_frames = len(vid_info['frames'])

        vid_data   = np.zeros((self.clip_length, self.max_objects, self.final_shape[0], self.final_shape[1], 3), dtype=np.float32)-1
        track_ids  = np.zeros((self.clip_length, self.max_objects), dtype=np.int32)-1
        obj_ids  = np.zeros((self.clip_length, self.max_objects),dtype=np.int64)-1
        labels     = np.zeros((self.clip_length, self.max_objects))-1
        bbox       = np.zeros((self.clip_length, self.max_objects, 4))-1
        bbox_head  = np.zeros((self.clip_length, self.max_objects, 4))-1
        key_pts    = np.zeros((self.clip_length, self.max_objects, self.num_keypoints, 3), dtype=np.float32)-1
        _key_pts    = np.zeros((self.clip_length, self.max_objects, self.num_keypoints, 3), dtype=np.float32)-1 #un-modified keypoints
        occlusions = np.zeros((self.clip_length, self.max_objects), dtype=np.int32)+1 #whether or not the entire pose is annotated
        centers    = np.zeros((self.clip_length, self.max_objects, 2))-1
        scales     = np.zeros((self.clip_length, self.max_objects, 2))-1
        inv_trans  = np.zeros((self.clip_length, self.max_objects, 2, 3))-1 #inverse affine transform matrix

        ignore_x   = [] #TODO: Set-up ignore regions for each frame. Variable-length lists
        ignore_y   = []

        target = np.zeros((self.clip_length, self.max_objects, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)-1
        target_weight = np.zeros((self.clip_length, self.max_objects, self.num_keypoints, 1), dtype=np.float32)-1

        frame_paths = []
        frame_ids   = np.zeros((self.clip_length), dtype=np.int64)
        for frame_ind in range(vid_frames):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            nframes        = frame['nframes'] #number of frames from original video
            frame_id       = frame['frame_id']
            vid_id         = frame['vid_id']
            is_labeled     = frame['is_labeled'] #frame contains at least one pose annotation
            categories     = frame['categories'][0]
            frame_index    = frame['frm_index'] #the nth annotated frame
            categories['skeleton'] = torch.Tensor(categories['skeleton'])

            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id 

            # Load frame, convert to RGB from BGR
            input_data = cv2.imread(os.path.join(base_path, frame_path))[...,::-1]
            #input_data = cv2.imread(os.path.join(base_path, frame_path))

            if self.load_type != 'test': #Test set does not include keypoint annotations
                # Extract bbox and label data from video info
                for obj_ind, obj in enumerate(frame['objs']):
                    trackid   = obj['trackid']
                    label     = obj['c'] #1: Only human class
                    occluded  = obj['occ'] #entire pose unannotated
                    obj_id    = obj['id'] #Concatenation of image id and obj track id
                    obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                    obj_head  = obj['bbox_head']
                    kpts      = obj['keypoints'] #17 points (x,y,valid)
                    #ignore_x  = obj['ignore_x'] #list of lists, combine x & y to create polygon region
                    #ignore_y  = obj['ignore_y']

                    kpts = np.array(kpts).reshape(self.num_keypoints, 3)
                    valid_kpts = kpts[kpts[:,-1] == 1]

                    '''
                    #bbox from keypoints 
                    xmin = np.min(valid_kpts[:,0])
                    xmax = np.max(valid_kpts[:,0])
                    ymin = np.min(valid_kpts[:,1])
                    ymax = np.max(valid_kpts[:,1])

                    w = (xmax - xmin)*1.15
                    h = (ymax - ymin)*1.15
                    '''

                    if obj_bbox == []:
                        obj_bbox = np.zeros(4)
                    
                    xmin, ymin, xmax, ymax = obj_bbox 

                    w = (xmax - xmin)
                    h = (ymax - ymin)
                    cx = xmin + w * 0.5
                    cy = ymin + h * 0.5

                    if w > self.aspect_ratio * h:
                        h = w * 1.0 / self.aspect_ratio
                    elif w < self.aspect_ratio * h:
                        w = h * self.aspect_ratio
                    scale = np.array(
                        [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                        dtype=np.float32)
                    if cx != -1:
                        scale = scale * self.sc

                    x1 = max(0, cx - (w/2))
                    x2 = min(width, cx + (w/2))
                    y1 = max(0, cy - (h/2))
                    y2 = min(height, cy + (h/2))

                    centers[frame_ind, obj_ind]      = [cx, cy]
                    scales[frame_ind, obj_ind]       = scale 
                    labels[frame_ind, obj_ind]       = label 
                    key_pts[frame_ind, obj_ind, :]   = kpts  
                    _key_pts[frame_ind, obj_ind, :]   = kpts  
                    track_ids[frame_ind, obj_ind]    = trackid
                    obj_ids[frame_ind, obj_ind]      = obj_id

                    occlusions[frame_ind, obj_ind]   = occluded
                    bbox[frame_ind, obj_ind, :]      = obj_bbox
                    bbox_head[frame_ind, obj_ind, :] = obj_head

                    mask = [True if o else False for o in key_pts[frame_ind,obj_ind,:,-1]]

                    temp_data, temp, out_params = self.transforms([input_data], {'center':centers[frame_ind, obj_ind], 'scale':scales[frame_ind, obj_ind],\
                            'key_pts':kpts[None,None,mask,:2], 'crop':[int(x1),int(y1),int(x2),int(y2)]})

                    inv_t = out_params['inv_trans']
                    flipped = out_params['flip']

                    temp_kpts = np.zeros((self.num_keypoints, 2))-1
                    temp_kpts[mask] = temp.squeeze()

                    vid_data[frame_ind, obj_ind] = temp_data 
                    if flipped:
                        for pair in self.flip_pairs:
                            temp_kpts[[pair[0],pair[1]]] = temp_kpts[[pair[1], pair[0]]]

                    key_pts[frame_ind, obj_ind, :, :2]  = temp_kpts 

                    inv_trans[frame_ind, obj_ind] = inv_t

                    obj_trgt, obj_trgt_wght = self.generate_target(key_pts[frame_ind, obj_ind])

                    target[frame_ind, obj_ind] = obj_trgt 
                    target_weight[frame_ind, obj_ind] = obj_trgt_wght

                    '''
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches

                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)

                    ax1.imshow(input_data)

                    rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                    ax1.add_patch(rect)
                    rect = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
                    ax1.add_patch(rect)
                    ax1.scatter(cx,cy)

                    ax2.imshow(temp_data[0].numpy())

                    plt.show()
                    '''

        vid_data       = torch.from_numpy(vid_data)
        key_pts        = torch.from_numpy(key_pts)
        bbox           = torch.from_numpy(bbox)
        bbox_head      = torch.from_numpy(bbox_head)
        target         = torch.from_numpy(target)
        target_weight  = torch.from_numpy(target_weight)

        #if number of available frames < requested frames, then repeat last frame
        if vid_frames < self.clip_length:
            vid_data[vid_frames:]          = vid_data[vid_frames-1]
            key_pts[vid_frames:]           = key_pts[vid_frames-1]
            bbox[vid_frames:]              = bbox[vid_frames-1]
            bbox_head[vid_frames:]         = bbox_head[vid_frames-1]
            target[vid_frames:]            = target[vid_frames-1]
            target_weight[vid_frames:]     = target_weight[vid_frames-1]

            frame_paths.extend(['None']*(self.clip_length - vid_frames)) #can't output non-string type None from DataLoader

        #Special case b/c of top-down detection, each object will have its own image
        #Combine frames and objects into same dimension
        vid_data = vid_data.view(self.clip_length*self.max_objects, self.final_shape[0], self.final_shape[1], 3)

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width)
        vid_data = vid_data.permute(3, 0, 1, 2)
        ret_dict = dict() 
        ret_dict['data']       = vid_data 

        annot_dict = dict()
        if self.viz:
            annot_dict['data']        = torch.Tensor(input_data).unsqueeze(0).permute(3,0,1,2) #Just for visualization 
            annot_dict['data']        = vid_data
        annot_dict['key_pts']     = key_pts
        annot_dict['_key_pts']     = _key_pts
        annot_dict['heatmaps']    = target
        annot_dict['heatmap_weights'] = target_weight 
        annot_dict['ignore_x']    = ignore_x
        annot_dict['ignore_y']    = ignore_y 
        annot_dict['bbox']        = bbox #bbox around pose 
        annot_dict['bbox_head']   = bbox_head #bbox around head 
        annot_dict['inv_trans']   = inv_trans 
        annot_dict['cls_labels']  = labels #class label. Always 1 for this dataset
        annot_dict['track_ids']   = track_ids 
        annot_dict['pose_occ']    = occlusions #pose level occlusion
        annot_dict['frame_paths'] = frame_paths  
        annot_dict['frame_ids']   = frame_ids    
        annot_dict['obj_ids']     = obj_ids 
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['stride']      = torch.Tensor(self.stride)
        annot_dict['nframes']     = nframes
        annot_dict['vid_id']      = vid_id
        annot_dict['categories']  = categories 
        annot_dict['load_type']   = self.load_type 
        annot_dict['frm_index']   = frame_index
        annot_dict['joint_names'] = self.joint_names 
        
        ret_dict['annots']     = annot_dict

        return ret_dict
