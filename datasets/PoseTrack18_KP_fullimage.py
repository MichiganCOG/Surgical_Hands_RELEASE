#Return keypoints of ground truth poses rather than image
import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class PoseTrack18_KP_fullimage(DetectionDataset):
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

    PoseTrack17 layout
    0 - Right Ankle 
    1 - Right Knee 
    2 - Right Hip 
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Head-bottom
    13 - Nose
    14 - Head-top
    """
    def __init__(self, *args, **kwargs):
        super(PoseTrack18_KP_fullimage, self).__init__(*args, **kwargs)

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

        self.num_keypoints = 17 #17 annotated body keypoints (but really 15)
        self.flip_pairs = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
        self.neighbor_link = [(10,8), (8,6), (9,7), (7,5),
                              (15,13), (13,11), (16,14), (14,12), (12,11), (11,5),
                              #(12,6), (6,1), (5,1), (1,0), (0,2),(0,4), (0,3)]
                              (12,6), (6,1), (5,1), (1,0), (0,2)]
        
        self.posetrack18_to_posetrack17 = [[0,13], [1,12], [2,14], [5,9], [6,8],
                                           [7,10], [8,7], [9,11], [10,6], [11,3],
                                           [12,2], [13,4], [14,1], [15,5], [16,0]]

        self.sample_all_obj = kwargs['sample_all_obj']
        self.aspect_ratio = self.image_width / self.image_height
        self.pixel_std = 200 

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

        self.feat_dim = 64
        self.feat_dir = os.path.join(kwargs['save_feat_dir'], 'FlowTrack-conv1_64_no_transform', self.load_type)

        for idx, item in enumerate(self.samples):
            segment_samples = {}
            
            for frm in item['frames']:
                if not frm['is_labeled']: #skip unlabeled frames
                    continue
                for obj in frm['objs']:
                    if obj['occ']: #skip un-annotated persons
                        continue 

                    kpts = np.array(obj['keypoints']).reshape(self.num_keypoints, 3)
                    if np.sum(kpts[:,-1]) < 1: #skip if ALL keypoints are occluded or un-annotated, and if only 1 keypoint exists
                        continue 

                    trackid = obj['trackid']
                    obj_id  = obj['id']

                    #Add saved feature map location
                    feat_path = os.path.join(self.feat_dir, frm['vid_id']+'.pkl')

                    f = [{'objs':[obj], 'img_path':frm['img_path'], 'feat_path':feat_path,\
                            'nframes':frm['nframes'], 'frame_id':frm['frame_id'],\
                            'vid_id':frm['vid_id'], 'is_labeled':frm['is_labeled'],\
                            'categories':frm['categories']}]

                    if trackid in segment_samples:
                        new_item = segment_samples[trackid]
                        new_item['frames'].extend(f)

                    else:
                        new_item = item.copy()
                        new_item['frames'] = f

                        if self.sample_all_obj:
                            segment_samples[str(obj_id)] = new_item
                        else:
                            segment_samples[trackid] = new_item

                if len(new_item['frames']) > max_clip_length:
                    max_clip_length = len(new_item['frames'])

            self.new_samples.extend(segment_samples.values())
        self.clip_length = max_clip_length
        self.samples = self.new_samples
        del self.new_samples

        print('Max clip length: {}'.format(max_clip_length))
        print('{} annotated objects in {}'.format(len(self.samples), self.load_type))

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path  = vid_info['base_path']
        vid_size   = vid_info['frame_size']

        input_data = []
        vid_data   = np.zeros((self.clip_length, self.feat_dim), dtype=np.float32)-1
        vid_data_pair = np.zeros((self.clip_length, self.feat_dim), dtype=np.float32)-1
        track_ids  = np.zeros(self.clip_length, dtype=np.int32)-1
        obj_ids  = np.zeros(self.clip_length, dtype=np.int64)-1
        labels     = np.zeros(self.clip_length)-1
        bbox       = np.zeros((self.clip_length, 4))-1
        bbox_head  = np.zeros((self.clip_length, 4))-1
        key_pts    = np.zeros((self.clip_length, self.num_keypoints, 3), dtype=np.float32)-1
        occlusions = np.zeros((self.clip_length), dtype=np.int32)+1 #whether or not the entire pose is annotated
        ignore_x   = [] #TODO: Set-up ignore regions for each frame. Variable-length lists
        ignore_y   = []

        frame_paths = []
        frame_ids   = np.zeros((self.clip_length), dtype=np.int64)

        for frame_ind in range(len(vid_info['frames'])):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            nframes        = frame['nframes']
            frame_id       = frame['frame_id']
            vid_id         = frame['vid_id']
            is_labeled     = frame['is_labeled'] #frame contains at least one pose annotation
            categories     = frame['categories']
            
            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id 

            if self.load_type != 'test': #Test set does not include keypoint annotations
                # Extract bbox and label data from video info
                if not is_labeled: #TODO: Find a way to deal with unlabeled frames, eventually
                    continue 

                vid_feat = torch.load(frame['feat_path']) #load saved features and transformed scaled key_pts
                oids     = vid_feat['object_ids'].squeeze() 
                trck_ids = vid_feat['track_id'].squeeze()

                for obj_ind, obj in enumerate(frame['objs']):
                    trackid   = obj['trackid']
                    label     = obj['c'] #1: Only human class
                    occluded  = obj['occ'] #entire pose unannotated
                    obj_id    = obj['id'] #Concatenation of image id and obj track id
                    obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                    #obj_head  = obj['bbox_head']
                    #kpts      = obj['keypoints'] #17 points (x,y,valid)
                    #ignore_x  = obj['ignore_x'] #list of lists, combine x & y to create polygon region
                    #ignore_y  = obj['ignore_y']

                    idx = (oids == obj_id).nonzero().item()

                    feat       = vid_feat['feat'][idx]
                    gt_key_pts = vid_feat['gt_key_pts'][idx]
                    
                    labels[frame_ind]       = label 
                    key_pts[frame_ind, :]   = gt_key_pts
                    track_ids[frame_ind]    = trackid
                    obj_ids[frame_ind]      = obj_id

                    occlusions[frame_ind]   = occluded

                    #Randomly select another sample
                    # 50% - same object within the same video, adjacent frame
                    # 40% - different object within the same video
                    # 10% - different object from different video
                    p = np.random.uniform()
                    uniq_tracks = len(set(trck_ids.tolist())) #unique objects in video - 2nd option isn't possible if only 1
                    num_occur   = torch.sum(trck_ids == trackid) #number of occurences of object - 3rd option isn't possible if only 1

                    #Select track from adjacent frame (forwards or backwards at random)
                    if (p < 0.5 and uniq_tracks > 1) or num_occur < 2:
                        pair_label = 0
                        indices = (trck_ids != trackid).nonzero()
                    else:
                        pair_label = 1
                        indices = (trck_ids == trackid).nonzero()

                    dist     = abs(indices - idx)
                    rand_idx = (dist.squeeze() == min(dist[dist>0])).nonzero()
                    perm     = torch.randperm(len(rand_idx))
                    rand_idx = indices[rand_idx[perm][0]].item()

                    '''
                    rand_idx = np.random.randint(len(oids))
                    oid = oids[rand_idx]
                    tid = trck_ids[rand_idx]

                    if p < 0.1:#diff object, diff vid
                        rand_idx = np.random.randint(len(self.samples))
                        sample = self.samples[rand_idx]
                        sample_vid_id   = sample['frames'][0]['vid_id']
                        sample_track_id = sample['frames'][0]['objs'][0]['trackid']

                        pair_label = 0
                        while rand_idx == idx or sample_vid_id == vid_id: 
                            rand_idx = np.random.randint(len(self.samples))
                            sample = self.samples[rand_idx]

                            sample_vid_id   = sample['frames'][0]['vid_id']
                            sample_track_id = sample['frames'][0]['objs'][0]['trackid']

                        vid_feat = torch.load(sample['frames'][0]['feat_path'])
                        rand_idx = np.random.randint(len(vid_feat['object_ids']))

                        oid = vid_feat['object_ids'][rand_idx].squeeze()

                    elif (p < 0.5 and uniq_tracks > 1) or num_occur < 2: #diff object, same vid
                        pair_label = 0

                        while tid == trackid or oid == obj_id:
                            rand_idx = np.random.randint(len(oids))
                            oid = oids[rand_idx]
                            tid = trck_ids[rand_idx]

                    else: #same object, different frame
                        pair_label = 1

                        while tid != trackid or oid == obj_id:
                            rand_idx = np.random.randint(len(oids))
                            oid = oids[rand_idx]
                            tid = trck_ids[rand_idx]
                    '''
                    
                    feat_pair    = vid_feat['feat'][rand_idx]
                    key_pts_pair = vid_feat['gt_key_pts'][rand_idx]

            del vid_feat 

            vid_data[frame_ind] = feat.squeeze()
            vid_data_pair[frame_ind] = feat_pair.squeeze()

        vid_data      = torch.from_numpy(vid_data)
        vid_data_pair = torch.from_numpy(vid_data_pair)
        key_pts       = torch.from_numpy(key_pts)
        pair_label    = torch.tensor(pair_label)

        #Normalize coordinates to be [0,1]?
        #key_pts      = key_pts/torch.tensor([self.image_width, self.image_height, 1.])
        #key_pts_pair = key_pts_pair/torch.tensor([self.image_width, self.image_height, 1.])

        '''
        print('pair: {}'.format(pair_label))
        frame1 = 255*np.ones((int(self.image_width), int(self.image_height)), dtype=np.uint8)
        frame2 = 255*np.ones((int(self.image_width), int(self.image_height)), dtype=np.uint8)
        for p1,p2 in self.neighbor_link:
            point1 = (int(key_pts[0,p1,0].item()),int(key_pts[0,p1,1].item()))
            point2 = (int(key_pts[0,p2,0].item()),int(key_pts[0,p2,1].item()))

            if point1[0] != -1 and point2[0] != -1:
                cv2.line(frame1, point1, point2, (0, 255, 0), 3)

            point1 = (int(key_pts_pair[0,p1,0].item()),int(key_pts_pair[0,p1,1].item()))
            point2 = (int(key_pts_pair[0,p2,0].item()),int(key_pts_pair[0,p2,1].item()))

            if point1[0] != -1 and point2[0] != -1:
                cv2.line(frame2, point1, point2, (0, 0, 100), 3)
        
        frame = np.hstack((frame1,frame2))
        cv2.imshow("Output-Keypoints",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        ret_dict = dict() 
        ret_dict['data']       = [vid_data, key_pts[...,:2], vid_data_pair, key_pts_pair[...,:2]]
        annot_dict = dict()
        annot_dict['pair_label']  = pair_label
        annot_dict['key_pts']     = key_pts
        annot_dict['ignore_x']    = ignore_x
        annot_dict['ignore_y']    = ignore_y 
        annot_dict['bbox']        = bbox #bbox around pose 
        annot_dict['bbox_head']   = bbox_head #bbox around head 
        annot_dict['cls_labels']  = labels #class label. Always 1 for this dataset
        annot_dict['track_ids']   = track_ids 
        annot_dict['pose_occ']    = occlusions #pose level annotation
        annot_dict['frame_paths'] = frame_paths  
        annot_dict['frame_ids']   = frame_ids    
        annot_dict['obj_ids']     = obj_ids 
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['nframes']     = nframes
        annot_dict['vid_id']      = vid_id
        annot_dict['categories']  = categories 
        
        ret_dict['annots']     = annot_dict

        return ret_dict
