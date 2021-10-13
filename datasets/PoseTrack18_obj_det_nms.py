#All objects detections on each frame

import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class PoseTrack18_obj_det_nms(DetectionDataset):
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
        super(PoseTrack18_obj_det_nms, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']
        self.json_path = kwargs['json_path']

        self.max_objects   = 1 #Unused only 1 object per output sample
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

        self.neighbor_link = [[10,8], [8,6], [9,7], [7,5],
                              [15,13], [13,11], [16,14], [14,12], [12,11], [11,5],
                              [12,6], [6,1], [5,1], [1,0], [0,2]]

        self.colors        = [[187,38,26],[187,38,26],[187,38,26],[187,38,26],
                              [172,201,63],[172,201,63],[172,201,63],[172,201,63],[172,201,63],
                              [92,200,97],[92,200,97],[92,200,97],[92,200,97],[28,84,197],[149,40,197]]

        self.joint_names = ['Nose', 'Head b', 'Head t', 'L Ear', 'R Ear', 'L Shoulder', 'R Shoulder',\
                            'L elbow', 'R elbow', 'L wrist', 'R wrist', 'L hip', 'R hip',\
                            'L knee', 'R knee', 'L ankle', 'R ankle']

        self.viz = kwargs['viz']

        self.det_threshold = kwargs.get('det_threshold', 0.0) #Object detection conf threshold, default to 0
        self.ignore_reg_threshold = kwargs.get('ignore_reg_threshold', 0.05) #Filter out detections w/ IOU with ignore region
        
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        self.new_samples = []
        #Get clip length, incase it was set to -1 in config
        max_clip_length = 0

        #Treat each detected object as a separate sample
        #This reserves memory and allows stacking of inputs
        for idx, item in enumerate(self.samples):

            segment_samples = {}
            for frm in item['frames']:
                if kwargs.get('labeled_frames_only', False):
                    if not frm['is_labeled']: #skip unlabeled frames
                        continue

                filter_vid_list = ['012834']
                #if self.load_type == 'val' and not frm['vid_id'] in filter_vid_list:
                #    continue

                objs = frm['objs']

                if len(objs) > 0:
                    ignore_x = objs[0]['ignore_x']
                    ignore_y = objs[0]['ignore_y']

                    pol_bboxes = []
                    for polx, poly in zip(ignore_x, ignore_y):
                        if polx == [] or poly == []:
                            continue 

                        xy = np.stack((np.array(polx),np.array(poly)),axis=1)

                        xmin, ymin = np.min(xy, axis=0)
                        xmax, ymax = np.max(xy, axis=0)
                        pol_bboxes.append([xmin, ymin, xmax, ymax])

                    pol_bboxes = np.array(pol_bboxes, dtype=np.float32)

                bboxes = []
                scores = []
                remaining_objs = []
                for obj in objs:

                    trackid = obj['trackid']
                    obj_id  = obj['id']

                    cls_score = obj.get('cls_score', 1.0) #default to 1.0
                    if cls_score < self.det_threshold:
                        continue 

                    #Skip detections with overlap with ignore regions
                    if len(pol_bboxes) > 0 and not self.json_path.rstrip('/') == '/z/home/natlouis/data/posetrack_data/annotations':
                        overlap, ind = IOU().iou(torch.tensor(obj['bbox']), torch.tensor(pol_bboxes))

                        if overlap.item() > self.ignore_reg_threshold:
                            continue

                    if obj['bbox'] == []: #when using GT dets, some bbox are blank
                        continue 
                    
                    xmin, ymin, xmax, ymax = obj['bbox']
                    area = (xmax-xmin)*(ymax-ymin)
                    #if area < 2500: #Skip small bounding boxes, typically can't be localized well
                    #    continue 

                    bboxes.append(obj['bbox'])
                    scores.append(cls_score)
                    remaining_objs.append(obj)

                if len(remaining_objs) == 0:
                    continue 
                
                #Apply NMS on the remaining bounding boxes
                keep = py_cpu_nms(np.array(bboxes), np.array(scores), kwargs.get('nms_threshold', 0.9))
                bboxes = []
                scores = []
                for idx in keep:
                    obj = remaining_objs[idx] 

                    trackid = obj['trackid']
                    obj_id  = obj['id']

                    f = [{'objs':[obj], 'img_path':frm['img_path'], \
                            'nframes':frm['nframes'], 'frame_id':frm['frame_id'],\
                            'vid_id':frm['vid_id'], 'is_labeled':frm['is_labeled'],\
                            'categories':frm['categories']}]

                    new_item = item.copy()
                    new_item['frames'] = f 

                    segment_samples[str(obj_id)] = new_item 

                    cls_score = obj.get('cls_score', 1.0) #default to 1.0

                    bboxes.append(obj['bbox'])
                    scores.append(cls_score)

                '''
                if self.load_type == 'val':
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches

                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    frame_path = frm['img_path']
                    input_data = cv2.imread(frame_path)[...,::-1]
                    ax1.imshow(input_data)

                    for polx, poly in zip(ignore_x, ignore_y):
                        xy = np.stack((np.array(polx),np.array(poly)),axis=1)
                        polygon = patches.Polygon(xy, closed=True)

                        ax1.add_patch(polygon)


                    for bbox,score in zip(bboxes,scores):
                        xmin, ymin, xmax, ymax = bbox
                        rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                        ax1.add_patch(rect)

                        area = int(np.sqrt((xmax-xmin)*(ymax-ymin))) #sqrt of area, just to limit digits
                        ax1.text(xmin, ymin, str(round(score,3)))

                    plt.show()
                '''

                if len(new_item['frames']) > max_clip_length:
                    max_clip_length = len(new_item['frames'])

            self.new_samples.extend(segment_samples.values())

        self.clip_length = max_clip_length
        self.samples = self.new_samples
        del self.new_samples

        print('Max clip length: {}'.format(max_clip_length))

        print('{} samples in {}'.format(len(self.samples), self.load_type))

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path  = vid_info['base_path']
        vid_size   = vid_info['frame_size']
        vid_frames = len(vid_info['frames'])

        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3), dtype=np.float32)-1
        track_ids  = np.zeros(self.clip_length, dtype=np.int32)-1
        target     = np.zeros((self.clip_length, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]))
        gt_key_pts = np.zeros((self.clip_length, self.num_keypoints, 3), dtype=np.float32)-1
        labels     = np.zeros(self.clip_length)-1
        bbox       = np.zeros((self.clip_length, 4))-1
        input_crop = np.zeros((self.clip_length, 4))-1

        centers    = np.zeros((self.clip_length, 2))-1
        scales     = np.zeros((self.clip_length, 2))-1
        trans      = np.zeros((self.clip_length, 2, 3))-1 #affine transform matrix
        inv_trans  = np.zeros((self.clip_length, 2, 3))-1 #inverse affine transform matrix
        cls_scores = np.zeros(self.clip_length)-1 

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
            categories['skeleton'] = torch.Tensor(categories['skeleton'])

            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id 

            # Load frame, convert to RGB from BGR
            input_data = cv2.imread(os.path.join(base_path, frame_path))[...,::-1]

            #Keep as BGR
            #input_data = cv2.imread(os.path.join(base_path, frame_path))

            if self.load_type != 'test': #Test set does not include keypoint annotations
                # Extract bbox and label data from video info
                obj = frame['objs'][0] #NOTE: Only one object
                trackid   = obj['trackid']
                label     = obj['c'] #1: Only human class
                obj_id    = obj['id'] #Concatenation of image id and obj track id
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                #kpts      = obj['keypoints'] #17 points (x,y,valid)
                #ignore_x  = obj['ignore_x'] #list of lists, combine x & y to create polygon region
                #ignore_y  = obj['ignore_y']
                label     = obj.get('cls_pred', 0) #default to 0
                cls_score = obj.get('cls_score', 1.0) #default to 1.0

                #del ignore_x
                #del ignore_y
                #kpts = np.array(kpts).reshape(self.num_keypoints, 3)
                #gt_key_pts[frame_ind] = kpts

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

                centers[frame_ind]      = [cx, cy]
                scales[frame_ind]       = scale 
                labels[frame_ind]       = label 
                track_ids[frame_ind]    = trackid
                cls_scores[frame_ind]   = cls_score 

                bbox[frame_ind, :]       = obj_bbox
                input_crop[frame_ind, :] = np.array([int(x1),int(y1),int(x2),int(y2)])

                #mask = [True if o else False for o in key_pts[frame_ind,:,-1]]

                temp_data, temp, out_params = self.transforms([input_data], {'center':centers[frame_ind], 'scale':scales[frame_ind],\
                        'key_pts':[], 'crop':[int(x1),int(y1),int(x2),int(y2)]})

                reg_t = out_params['trans']
                inv_t = out_params['inv_trans']
                flipped = out_params['flip']

                vid_data[frame_ind] = temp_data
                trans[frame_ind]     = reg_t
                inv_trans[frame_ind] = inv_t

                '''
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.imshow(input_data)

                xmin, ymin, xmax, ymax = bbox[0]
                rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)

                ax1.text(xmin, ymin, str(cls_scores[0])[:6])

                for polx, poly in zip(ignore_x, ignore_y):
                    xy = np.stack((np.array(polx),np.array(poly)),axis=1)
                    polygon = patches.Polygon(xy, closed=True)

                    ax1.add_patch(polygon)

                plt.show()
                '''

        vid_data       = torch.from_numpy(vid_data)
        bbox           = torch.from_numpy(bbox)
        trans          = torch.from_numpy(trans)
        inv_trans      = torch.from_numpy(inv_trans)

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width)
        vid_data = vid_data.permute(3, 0, 1, 2)
        ret_dict = dict() 
        ret_dict['data']       = vid_data 

        annot_dict = dict()
        #annot_dict['data']        = np.copy(input_data) #For visualization purposes only
        if self.viz:
            annot_dict['data']        = np.copy(input_data) #For visualization purposes only
            #annot_dict['data']        = vid_data
        #annot_dict['gt_key_pts']    = gt_key_pts 
        annot_dict['heatmaps']      = target
        annot_dict['bbox']          = bbox #bbox around pose 
        annot_dict['input_crop']    = input_crop 
        annot_dict['trans']         = trans 
        annot_dict['inv_trans']     = inv_trans 
        annot_dict['cls_labels']    = labels #class label. Always 1 for this dataset
        annot_dict['frame_path']    = frame_paths  
        annot_dict['frame_ids']     = frame_ids    
        annot_dict['frame_size']    = vid_size #width, height
        annot_dict['raw_frame_size'] = np.array(vid_size) #width, height
        annot_dict['stride']        = torch.Tensor(self.stride)
        annot_dict['nframes']       = nframes
        annot_dict['vid_id']        = vid_id
        annot_dict['categories']    = categories 
        annot_dict['load_type']     = self.load_type 
        annot_dict['neighbor_link'] = torch.tensor(self.neighbor_link)
        annot_dict['link_colors']   = torch.tensor(self.colors)
        annot_dict['joint_names'] = self.joint_names 
        
        ret_dict['annots']     = annot_dict

        return ret_dict

class IOU():
    """
    Intersection-over-union between one prediction bounding box 
    and plausible ground truth bounding boxes

    """
    def __init__(self, *args, **kwargs):
        pass

    def intersect(self, box_p, box_t):
        """
        Intersection area between predicted bounding box and 
        all ground truth bounding boxes

        Args:
            box_p (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            box_t (Tensor, shape [N,4]): target bounding boxes

        Return:
            intersect area (Tensor, shape [N]): intersect_area for all target bounding boxes
        """
        x_left = torch.max(box_p[0], box_t[:,0])
        y_top = torch.max(box_p[1], box_t[:,1])
        x_right = torch.min(box_p[2], box_t[:,2])
        y_bottom = torch.min(box_p[3], box_t[:,3])

        width = torch.clamp(x_right - x_left, min=0)
        height = torch.clamp(y_bottom - y_top, min=0)

        intersect_area = width * height

        return intersect_area

    def iou(self, box_p, box_t):
        """
        Performs intersection-over-union 

        Args:
            box_p (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            box_t (Tensor, shape [N,4]): target bounding boxes

        Return:
            overlap (Tensor, shape [1]): max overlap
            ind     (Tensor, shape [1]): index of bounding box with largest overlap
        """

        intersect_area = self.intersect(box_p, box_t)

        box_p_area = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
        box_t_area = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1])
        union = box_p_area + box_t_area - intersect_area

        #NOTE: Make sure to remove this line. It's only to get around area of single keypoints, for now
        union = torch.clamp(union, min=0.001)

        overlap = torch.max(intersect_area/union)
        ind     = torch.argmax(intersect_area/union)

        assert overlap >= 0.0
        assert overlap <= 1.0

        return overlap, ind

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
