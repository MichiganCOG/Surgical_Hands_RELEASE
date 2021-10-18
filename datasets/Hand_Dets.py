#Run forward on novel hand videos (no annotations) on hand crops
#Assume NMS has already been applied to the detections

import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import cv2
import os
import numpy as np

class Hand_Dets(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(Hand_Dets, self).__init__(*args, **kwargs)

        self.load_type    = kwargs['load_type']

        self.max_objects  = 1
        self.final_shape  = kwargs['final_shape']
        self.heatmap_size = kwargs['heatmap_size']
        self.image_height = self.final_shape[0]
        self.image_width  = self.final_shape[1]
        self.stride       = (self.image_width/self.heatmap_size[0], \
                             self.image_height/self.heatmap_size[1])
        self.num_keypoints = 21 #21 annotated hand keypoints

        self.sc = kwargs['sc']
        
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

        self.joint_names = ['Wrist', 'Thumb_k', 'Thumb_b', 'Thumb_m', 'Thumb_t', \
                               'Index_k', 'Index_b', 'Index_m', 'Index_t', \
                               'Middle_k', 'Middle_b', 'Middle_m', 'Middle_t', \
                               'Ring_k', 'Ring_b', 'Ring_m', 'Ring_t', \
                               'Pinky_k', 'Pinky_b', 'Pinky_m', 'Pinky_t']

        self.viz = kwargs['viz']

        self.det_threshold = kwargs.get('det_threshold', 0.0) #Object detection conf threshold, default to 0
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

        self.new_samples = []
        max_clip_length = 0
        print('Number of samples: {} in {}'.format(len(self.samples), self.load_type))

        #Treat each detected object as a separate sample
        save_feat_dir = os.path.join(kwargs['save_feat_dir'], kwargs['model']+'-'+kwargs['exp'])
        for idx, item in enumerate(self.samples):

            filter_vid_list = ['9DhftZeReiI_000008_000720']
            vid_id = item['frames'][0]['vid_id']

            #if vid_id not in filter_vid_list:
            #    continue 

            segment_samples = {}
            frames = item['frames']
            width, height = item['frame_size']
            for frm in frames:
                objs = frm['objs']
                bboxes = []
                scores = []

                img_path = frm['img_path']
                frame_id = img_path.split('/')[-1].split('.')[0]

                if not frm['is_labeled']: #only evaluate on labeled frames
                    continue 

                for obj_ind, obj in enumerate(objs):
                    new_item = item.copy()

                    cls_score = obj.get('cls_score', 1.0) #default to 1.0
                    if cls_score < self.det_threshold:
                        continue

                    if obj['bbox'] == []: #incase no detection
                        continue

                    xmin, ymin, xmax, ymax = obj['bbox']
                    #area = (xmax-xmin)*(ymax-ymin)
                    #if area < 1000: #Skip small bounding boxes, typically can't be localized well
                    #    continue

                    bboxes.append(obj['bbox'])
                    scores.append(cls_score)
                    f = [{'objs':[obj], 'img_path':img_path, 'frame_id':frm['frame_id'], 'vid_id':frm['vid_id']}]

                    new_item['frames'] = f

                    obj_id = '_'.join((vid_id, frame_id, str(obj_ind)))
                    segment_samples[obj_id] = new_item

                if len(new_item['frames']) > max_clip_length:
                    max_clip_length = len(new_item['frames'])

                '''
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig = plt.figure()
                ax = fig.add_subplot(111)

                base_path  = item['base_path']
                frame_path = frm['img_path']
                vis = (cv2.imread(os.path.join(base_path, frame_path))[...,::-1])
                plt.imshow(vis)

                for bbox, score in zip(bboxes, scores):
                    xmin, ymin, xmax, ymax = bbox
                    rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)

                    ax.text(xmin, ymin, str(score))

                plt.show()
                '''

            self.new_samples.extend(segment_samples.values())

        self.clip_length = max_clip_length
        self.samples = self.new_samples 
        del self.new_samples

        print('Number of filtered samples: {} in {}'.format(len(self.samples), self.load_type))

    def __getitem__(self, idx):
        vid_info  = self.samples[idx]
        base_path = vid_info['base_path']

        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3), dtype=np.float32)-1
        bbox       = np.zeros((self.clip_length, 4))-1
        target     = np.zeros((self.clip_length, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]))
        hand_crops = np.zeros((self.clip_length, 4))-1
        padding    = np.zeros((self.clip_length, 4), dtype=np.int32)
        conf_score = np.zeros((self.clip_length))-1
        cls_score  = np.zeros((self.clip_length))-1
        labels     = np.zeros((self.clip_length))-1
    
        frame_ids   = np.zeros((self.clip_length), dtype=np.int64)
        frame_paths = []
        for frame_ind in range(len(vid_info['frames'])):
            frame         = vid_info['frames'][frame_ind]
            frame_path    = vid_info['frames'][frame_ind]['img_path']
            frame_id      = int(frame['frame_id'])
            vid_id         = frame['vid_id']

            frame_paths.append(frame_path)
            frame_ids[frame_ind] = frame_id
            #frame_idx     = vid_info['frames'][frame_ind]['frame_idx']
            width, height = vid_info['frame_size']

            # Load frame image data and preprocess image accordingly
            input_data = cv2.imread(frame_path)[...,::-1]

            obj = frame['objs'][0] #NOTE: Only one object
            bbox[frame_ind]       = obj['bbox']
            conf_score[frame_ind] = obj['confidence']
            cls_score[frame_ind]  = obj['cls_score']
            labels[frame_ind]     = obj['cls_pred']
            xmin, ymin, xmax, ymax = bbox[frame_ind]

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
            if xmax > width:
                pr = abs(width - xmax) 
            if ymax > height:
                pb = abs(height - ymax)

            hand_crop = [xmin+pl, ymin+pt, xmax, ymax]

            hand_crops[frame_ind] = hand_crop 
            padding[frame_ind]    = [pl, pt, pr, pb]

            # Preprocess data
            temp_data, temp, _  = self.transforms(cv2.copyMakeBorder(input_data, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)[None], {'bbox_data':np.zeros((self.clip_length,1,21,2)), 'hand_crop':hand_crop, 'label':labels})

            vid_data[frame_ind] = temp_data 

            '''
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.imshow(input_data)

            for ind in range(self.max_objects):
                xmin, ymin, xmax, ymax = bbox[ind]
                rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)

                ax1.text(xmin, ymin, str(cls_score[ind])[:6])

                xmin, ymin, xmax, ymax = hand_crops[ind]
                rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
            plt.show()
            '''

        vid_data   = torch.from_numpy(vid_data)
        labels     = torch.from_numpy(labels).float()

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict           = dict() 
        ret_dict['data']   = vid_data 

        annot_dict               = dict()
        if self.viz:
            annot_dict['data']   = cv2.cvtColor(np.copy(input_data), cv2.COLOR_BGR2RGB)
        annot_dict['labels']     = labels
        annot_dict['heatmaps']   = target
        annot_dict['bbox']       = bbox 
        annot_dict['conf_score'] = conf_score
        annot_dict['input_crop'] = hand_crops  
        annot_dict['padding']    = padding 
        #annot_dict['frame_idx']  = frame_idx
        annot_dict['frame_path'] = frame_paths 
        annot_dict['frame_ids']  = frame_ids
        annot_dict['frame_size'] = (width, height) 
        annot_dict['raw_frame_size'] = np.array([width, height])
        annot_dict['nframes']    = 1 #not useful
        annot_dict['vid_id']     = frame['vid_id']
        annot_dict['neighbor_link'] = torch.tensor(self.neighbor_link)
        annot_dict['link_colors']   = torch.tensor(self.colors)
        annot_dict['joint_names']   = self.joint_names 
        annot_dict['categories']    = self.categories 

        ret_dict['annots']   = annot_dict

        return ret_dict
