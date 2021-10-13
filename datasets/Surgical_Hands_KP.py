#Return keypoints of ground truth poses rather than image
import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class Surgical_Hands_KP(DetectionDataset):
    """
    Data annotated from publicly available surgical hand videos

    x training samples
    x testing samples
    """
    def __init__(self, *args, **kwargs):
        super(Surgical_Hands_KP, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']
        self.json_path = kwargs['json_path']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects   = 1
        self.sigma         = kwargs['gaussian_sigma']
        self.heatmap_size  = kwargs['heatmap_size']
        self.image_height = self.final_shape[0]
        self.image_width  = self.final_shape[1]
        self.stride        = (self.image_width/self.heatmap_size[0], \
                              self.image_height/self.heatmap_size[1]) #effective stride of the network

        self.num_keypoints = 21 #21 annotated hand keypoints

        self.sc = kwargs['sc']
        self.mask_occ = False 

        self.in_channels = kwargs['in_channels'] # 2: (x,y), 3: (x,y,confidence)

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

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        #Remove items without keypoint annotations
        #And separate every ground truth object into a separate sample. 
        #This reserves memory and allows stacking of inputs
        print('{} videos in {}'.format(len(self.samples), self.load_type))

        self.new_samples = []
        max_clip_length = 0 

        tags = kwargs['tags'] #Expecting folda# as the first (and only) tag
        gcn_feat_dir  = kwargs['gcn_feat_dir']
        self.feat_dim = 64
        self.feat_dir = os.path.join(kwargs['save_feat_dir'], gcn_feat_dir+tags[0], self.load_type)

        self.new_samples = []
        for idx, item in enumerate(self.samples):
            width, height = item['frame_size']
            
            for frm in item['frames']:
                if not frm['is_labeled']: #skip unlabeled frames
                    continue

                for obj in frm['objs']:
                    kpts = np.array(obj['hand_pts']).reshape(self.num_keypoints, 3)
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

                    #Add saved feature map location
                    feat_path = os.path.join(self.feat_dir, frm['vid_id']+'.pkl')

                    new_item = {}
                    new_item['frames'] = [{'objs':[obj], 'img_path':frm['img_path'], 'feat_path':feat_path,\
                            'frame_id':frm['frame_id'], 'vid_id':frm['vid_id'], 'is_labeled':frm['is_labeled']}]

                    new_item['base_path']  = item['base_path']
                    new_item['frame_size'] = item['frame_size']

                    self.new_samples.append(new_item)

        self.samples = self.new_samples
        del self.new_samples

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
        unannotated = np.zeros((self.clip_length, 21), dtype=np.int32)-1 #21 keypoints
        ignore_x   = [] #TODO: Set-up ignore regions for each frame. Variable-length lists
        ignore_y   = []

        frame_paths = []
        frame_ids   = np.zeros((self.clip_length), dtype=np.int64)

        for frame_ind in range(len(vid_info['frames'])):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            frame_id       = frame['frame_id']
            vid_id         = frame['vid_id']
            
            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id 

            if self.load_type != 'test': #Test set does not include keypoint annotations
                # Extract bbox and label data from video info

                vid_feat = torch.load(frame['feat_path']) #load saved features and transformed scaled key_pts
                oids     = vid_feat['object_ids'].squeeze() 
                trck_ids = vid_feat['track_id'].squeeze()

                for obj_ind, obj in enumerate(frame['objs']):
                    trackid   = obj['trackid']
                    obj_id    = obj['id'] #Concatenation of image id and obj track id
                    obj_id    = int(''.join((obj_id.split('_')[-4:]))) #make numeric only
                    label     = 0 if obj['c'] == 'left' else 1 #0: left hand, 1: right hand
                    unann     = obj['occ'] #entire pose unannotated
                    obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                    #hand_pts      = obj['hand_pts'] #21 points (x,y,visibility)

                    idx = (oids == obj_id).nonzero().item()

                    feat       = vid_feat['feat'][idx]
                    gt_key_pts = vid_feat['gt_key_pts'][idx]
                    
                    visibility = 1 - np.array(unann)
                    if self.mask_occ:
                        for i, v in enumerate(visibility):
                            if v == 1:
                                unann[i] = True

                    labels[frame_ind]       = label 
                    key_pts[frame_ind, :]   = gt_key_pts
                    track_ids[frame_ind]    = trackid
                    obj_ids[frame_ind]      = obj_id

                    unannotated[frame_ind]   = unann

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
                    rand_idx = (dist.squeeze(dim=1) == min(dist[dist>0])).nonzero()
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
                    
                    feat_pair     = vid_feat['feat'][rand_idx]
                    key_pts_pair  = vid_feat['gt_key_pts'][rand_idx]
                    obj_bbox_pair = vid_feat['bbox'][rand_idx][0]

            del vid_feat 

            vid_data[frame_ind] = feat.squeeze()
            vid_data_pair[frame_ind] = feat_pair.squeeze()

        #Clip visibility values
        key_pts[...,2] = np.clip(key_pts[...,2], 0,1)
        key_pts_pair[...,2] = np.clip(key_pts_pair[...,2], 0,1)

        vid_data      = torch.from_numpy(vid_data)
        vid_data_pair = torch.from_numpy(vid_data_pair)
        key_pts       = torch.from_numpy(key_pts)
        pair_label    = torch.tensor(pair_label)

        obj_width  = obj_bbox[2]-obj_bbox[0]
        obj_height = obj_bbox[3]-obj_bbox[1]
        obj_pair_width  = obj_bbox_pair[2]-obj_bbox_pair[0]
        obj_pair_height = obj_bbox_pair[3]-obj_bbox_pair[1]

        #Normalize coordinates to be [0,1]?
        key_pts      = (key_pts-torch.tensor([obj_bbox[0], obj_bbox[1], 0.]))/torch.tensor([obj_width, obj_height, 1.])
        key_pts_pair = (key_pts_pair-torch.tensor([obj_bbox_pair[0], obj_bbox_pair[1], 0.]))/torch.tensor([obj_pair_width, obj_pair_height, 1.])

        #Zero-out unannotated points (i.e set to -1), but save "confidence" scores
        scores = torch.clone(key_pts[...,2])
        key_pts[scores == 0] = -1
        key_pts[...,2] = scores

        scores = torch.clone(key_pts_pair[...,2])
        key_pts_pair[scores == 0] = -1
        key_pts_pair[...,2] = scores

        '''
        print('obj 0:\n {}'.format(key_pts))
        print('obj 1:\n {}'.format(key_pts_pair))
        print('pair: {}'.format(pair_label))
        frame1 = 255*np.ones((600, 600, 3), dtype=np.uint8)
        frame2 = 255*np.ones((600, 600, 3), dtype=np.uint8)
        for (p1,p2),c in zip(self.neighbor_link, self.colors):
            point1 = (int(key_pts[0,p1,0].item() * obj_width),int(key_pts[0,p1,1].item() * obj_height))
            point2 = (int(key_pts[0,p2,0].item() * obj_width),int(key_pts[0,p2,1].item() * obj_height))

            if key_pts[0,p1,-1] != 0:
                cv2.circle(frame1, (point1[0], point1[1]), color=(c[2],c[1],c[0]), radius=5, thickness=-1)
            if key_pts[0,p2,-1] != 0:
                cv2.circle(frame1, (point2[0], point2[1]), color=(c[2],c[1],c[0]), radius=5, thickness=-1)
            if key_pts[0,p1,-1] != 0 and key_pts[0,p2,-1] != 0:
                cv2.line(frame1, point1, point2, (c[2], c[1], c[0]), 3)

            point1 = (int(key_pts_pair[0,p1,0].item() * obj_pair_width),int(key_pts_pair[0,p1,1].item() * obj_pair_height))
            point2 = (int(key_pts_pair[0,p2,0].item() * obj_pair_width),int(key_pts_pair[0,p2,1].item() * obj_pair_height))

            if key_pts_pair[0,p1,-1] != 0:
                cv2.circle(frame2, (point1[0], point1[1]), color=(c[2],c[1],c[0]), radius=5, thickness=-1)
            if key_pts_pair[0,p2,-1] != 0:
                cv2.circle(frame2, (point2[0], point2[1]), color=(c[2],c[1],c[0]), radius=5, thickness=-1)
            if key_pts_pair[0,p1,-1] != 0 and key_pts_pair[0,p2,-1] != 0:
                cv2.line(frame2, point1, point2, (c[2], c[1], c[0]), 3)
        
        frame = np.hstack((frame1,frame2))
        cv2.imshow("Output-Keypoints",frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        key_pts_pair = np.array(key_pts_pair, dtype=np.float32)
        ret_dict = dict() 
        if self.in_channels == 2: #coordinates only
            ret_dict['data']       = [vid_data, key_pts[...,:2], vid_data_pair, key_pts_pair[...,:2]]
        else:
            ret_dict['data']       = [vid_data, key_pts, vid_data_pair, key_pts_pair]
        annot_dict = dict()
        annot_dict['pair_label']  = pair_label
        annot_dict['key_pts']     = key_pts
        annot_dict['ignore_x']    = ignore_x
        annot_dict['ignore_y']    = ignore_y 
        annot_dict['bbox']        = bbox #bbox around pose 
        annot_dict['bbox_head']   = bbox_head #bbox around head 
        annot_dict['cls_labels']  = labels #class label. Always 1 for this dataset
        annot_dict['track_ids']   = track_ids 
        annot_dict['occ']         = unannotated
        annot_dict['frame_paths'] = frame_paths  
        annot_dict['frame_ids']   = frame_ids    
        annot_dict['obj_ids']     = obj_ids 
        annot_dict['frame_size']  = vid_size #width, height
        annot_dict['vid_id']      = vid_id
        annot_dict['categories']  = self.categories 
        
        ret_dict['annots']     = annot_dict

        return ret_dict
