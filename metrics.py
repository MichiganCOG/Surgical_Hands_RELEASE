import os
import sys
import subprocess
import json 
import cv2

import numpy as np
import torch
import torch.nn.functional as F

import wandb 

from models.gcn.gcn import GCN

import utils.flowlib as fl
from hungarian_algorithm.hungarian import Hungarian, HungarianError

class Metrics(object):
    def __init__(self, *args, **kwargs):
        """
        Compute accuracy metrics from this Metrics class
        Args:
            acc_metric (String): String used to indicate selected accuracy metric 
    
        Return:
            None
        """
        self.metric_type = kwargs['acc_metric'] 

        if self.metric_type == 'Accuracy':
            self.metric_object = Accuracy(*args, **kwargs) 
        elif self.metric_type == 'pck_curve_hand':
            self.metric_object = PCK_Curve_Hand(*args, **kwargs)
        elif self.metric_type == 'PCK_FlowTrack':
            self.metric_object = PCK_FlowTrack(*args, **kwargs)
        elif self.metric_type == 'Contrastive_Accuracy':
            self.metric_object = Contrastive_Accuracy(*args, **kwargs)
        elif self.metric_type == 'Save_Video_Keypoints':
            self.metric_object = Save_Video_Keypoints(*args, **kwargs)
        elif self.metric_type == 'Save_Frame_Video_Heatmaps':
            self.metric_object = Save_Frame_Video_Heatmaps(*args, **kwargs)
        elif self.metric_type == 'Eval_PoseTrack18_det':
            self.metric_object = Eval_PoseTrack18_det(*args, **kwargs)
        elif self.metric_type == 'Eval_PoseTrack17_det':
            self.metric_object = Eval_PoseTrack17_det(*args, **kwargs)
        else:
            self.metric_type = None

    def get_accuracy(self, predictions, targets, **kwargs):
        """
        Return accuracy from selected metric type

        Args:
            predictions: model predictions 
            targets: ground truth or targets 
        """

        if self.metric_type == None:
            return -1

        else:
            return self.metric_object.get_accuracy(predictions, targets, **kwargs)

class Accuracy(object):
    """
    Standard accuracy computation. # of correct cases/# of total cases

    """
    def __init__(self, *args, **kwargs):
        self.ndata = kwargs['ndata']
        self.count = 0
        self.correct = 0.
        self.total   = 0. 

    def get_accuracy(self, predictions, data):
        """
        Args:
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

        Return:
            Accuracy # of correct case/ # of total cases
        """
        targets = data['labels']
        assert (predictions.shape[0] == targets.shape[0])

        if self.count >= self.ndata:
            self.count   = 0
            self.correct = 0
            self.total   = 0

        targets     = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        if len(targets.shape) == 2 and len(predictions.shape) == 2:
            self.correct += np.sum(np.argmax(predictions,1) == targets[:, -1])
            self.total   += predictions.shape[0]

        else: 
            self.correct += np.sum(np.argmax(predictions,1) == targets[:, -1])
            self.total   += predictions.shape[0]

        # END IF

        self.count += predictions.shape[0]

        return self.correct/self.total

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

    def get_accuracy(self, prediction, targets):
        """
        Args:
            prediction (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            targets    (Tensor, shape [N,4]): target bounding boxes

        Return:
            iou (Tensor, shape[1]): Highest iou amongst target bounding boxes
            ind (Tensor, shape[1]): Index of target bounding box with highest score
        """

        iou_score, ind = self.iou(prediction, targets)
        return iou_score, ind

class PCK_Curve_Hand():
    def __init__(self, threshold=torch.linspace(0,1,101),**kwargs):
    #def __init__(self, threshold=torch.Tensor([0.5]),**kwargs):
        """"
        Probability of Correct Keypoint (PCK) evaluation metric for OpenPose hand model

        Args:
            threshold (Tensor, shape [11]): normalized distance thresholds 
            ndata                 (scalar): total number of datapoints in dataset

        Returns:
            None 
        """

        self.threshold = threshold
        self.acc = torch.zeros(len(self.threshold), 21)

        self.viz    = kwargs['viz']
        self.ndata  = kwargs['ndata']
        self.nbatch = np.ceil(self.ndata/kwargs['batch_size']).astype(np.int)
        self.count  = 0

        self.logger = kwargs['logger'] #wandb logging tool
        self.debug  = kwargs['debug']

    def get_accuracy(self, predictions, data):
        """
        Args:
            predictions: 
            data 
        """

        if self.count == self.nbatch: #reset accuracy each epoch
            self.count = 0
            self.acc = torch.zeros(len(self.threshold), 21)

        W,H = data['frame_size']
        hand_pts   = data['key_pts']
        occluded   = data['occ']
        padding    = data['padding']
        input_crop = data.get('input_crop', data['bbox'])

        if 'head_size' in data:
            #head_size = data['head_size'].float() / H.float() * 368
            head_size = data['head_size'].float()
            dist_thresh = 0.7 * head_size.unsqueeze(1) * self.threshold.repeat(len(head_size),1) #Distance threshold is normalized to 0.7*head_size
        else:
            #if no given head size, then normalize to longest side of bounding box
            bbox = data['bbox']
            bbox_size = torch.max(bbox[...,2]-bbox[...,0], bbox[...,3]-bbox[...,1])
            dist_thresh = bbox_size * self.threshold.repeat(len(bbox_size),1) #Distance threshold is normalized to longest side of bounding box 

        if isinstance(predictions, tuple):
            _, _, _, _, _, out6 = predictions
        else:
            out6 = predictions[:,0] 
        
        import matplotlib.pyplot as plt

        B,D,H_,W_ = out6.shape

        x = hand_pts[:,0,:,0]
        y = hand_pts[:,0,:,1]
        gt_pts = torch.stack((x,y)).permute(1,2,0).float()
        mask = (1 - data['occ'][:,0])[...,None]

        pred_pts = []
        for b in range(B):
            #kp = out6[b, :-1]: #ignore last layer (background)
            kp = out6[b]  #not regressing to background layer 

            max_indices = torch.argmax(kp.view(-1,H_*W_), dim=1)
            rows = (max_indices / W_).float()
            cols = (max_indices % H_).float()

            #adjust for padding and image crpo
            pl,pt,pr,pb = padding[b,0] 
            crop = input_crop[b,0]
            crop_h = crop[3]-crop[1] 
            crop_w = crop[2]-crop[0]

            '''
            #GT data 
            temp = data['temp']
            cols = temp[b,0,:,0]
            rows = temp[b,0,:,1]

            x_new = np.ceil((cols / 368 * crop_w) - pl + crop[0])
            y_new = np.ceil((rows / 368 * crop_h) - pt + crop[1])
            '''

            x_new = ((cols / W_ * crop_w) - pl + crop[0]).int()
            y_new = ((rows / H_ * crop_h) - pt + crop[1]).int()

            pred_pts.append(torch.stack((x_new,y_new), dim=1))
            
            if self.viz:
                B,H,W,C = data['data'].shape
                plt.figure(figsize=(16,7))
                extent = np.int(0), np.int(368), np.int(0), np.int(368)

                plt.subplot(2,3,1)
                img = data['data'][b].cpu().numpy()
                plt.imshow(img)
                plt.scatter(x[b],y[b],c='g')
                plt.scatter(x_new.cpu(), y_new.cpu(),c='r')
                plt.title('Image with keypoints')

                plt.subplot(2,3,2)
                #heatmap = torch.max(out6[0,:-1],dim=0)[0]
                heatmap = torch.max(out6[b],dim=0)[0]
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(heatmap.cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
                plt.title('Predicted')

                plt.subplot(2,3,3)
                #gt_heatmap = torch.max(data['heatmaps'][0,:-1],dim=0)[0]
                gt_heatmap = torch.max(data['heatmaps'][b,0],dim=0)[0]
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(gt_heatmap.cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
                plt.title('Groundtruth')

                plt.subplot(2,3,5)
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(out6[0,-1].cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
                plt.title('Predicted background')

                plt.subplot(2,3,6)
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(data['heatmaps'][0,-1].cpu().numpy(), cmap='jet', alpha=0.5, interpolation='none', extent=extent)
                plt.title('Background')

                plt.show()

        pred_pts = torch.stack(pred_pts).float().cpu()
        dist = torch.norm(pred_pts*mask-gt_pts, dim=2)

        for idx in range(len(self.threshold)):
            self.acc[idx, :] += torch.sum((dist <= dist_thresh[:,idx].unsqueeze(1)),dim=0,dtype=torch.float)/self.ndata

        self.count += 1

        if self.count == self.nbatch:
            if self.viz:
                plt.plot(self.threshold.cpu().numpy(), torch.mean(self.acc, dim=1).cpu().numpy())
                plt.grid(True, 'both')
                plt.xlabel('Normalized Distance')
                plt.ylabel('PCK')

                plt.ylim([0, 1])

                plt.show()

            if not self.debug:
                mean_acc = torch.mean(self.acc, dim=1).cpu().numpy()
                for idx in range(len(self.threshold)):
                    self.logger.log({'PCKh (Normalized Distance)':mean_acc[idx]})


        #Return area under the curve
        return torch.sum(torch.mean(self.acc, dim=1)/len(self.acc)).item()

#Mostly adapted from: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/
class PCK_FlowTrack():
    def __init__(self, threshold=0.5, **kwargs):
        """"
        Probability of Correct Keypoint (PCK) evaluation metric
        but uses ground truth heatmap rather than x,y locations

        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies

        Args:
            threshold (Tensor, shape [11]): normalized distance thresholds 
            ndata                 (scalar): total number of datapoints in dataset

        Returns:
            None 
        """

        self.viz = kwargs['viz']
        self.threshold = threshold

        self.ndata  = kwargs['ndata']
        self.nbatch = np.ceil(self.ndata/kwargs['batch_size']).astype(np.int)

        self.count   = 0
        self.correct = 0
        self.total   = 0

        self.logger = kwargs['logger'] #wandb logging tool 
        self.debug  = kwargs['debug'] 

    def get_max_preds(self, batch_heatmaps):
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

    def calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists, thr=0.5):
        #Let's return the number of dists vals below threshold and average across all data
        dist_cal = np.not_equal(dists, -1) 
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return np.less(dists[dist_cal], thr).sum(), num_dist_cal
        else:
            return 0,0

    def get_accuracy(self, predictions, data):
        #model may output intermediate feature maps in tuple
        if isinstance(predictions, tuple): 
            predictions = predictions[-1][:,:21].unsqueeze(1) #ignore last layer, unsqueeze T dim

        target = data['heatmaps'].numpy()
        predictions = predictions.cpu().numpy()

        B,T,D,H,W = target.shape

        #Reshape, temporal dimension now represents multiple objects per image
        target = np.reshape(target, (B*T,D,H,W))
        predictions = np.reshape(predictions, (B*T,D,H,W))
        idx = list(range(predictions.shape[1]))
        norm = 1.0 

        self.count += B

        if self.count > self.ndata:
            self.count   = 0
            self.correct = 0
            self.total   = 0
        
        if self.viz:
            joints = data['joint_names']

            import matplotlib.pyplot as plt
            for bt in range(B*T):
                b = int(bt/T)
                t = bt % T

                img = data['data'][b,:,t].permute(1,2,0).cpu().numpy()
                mean = np.array([[[123.675,116.28,103.52]]])
                std = np.array([[[58.395,57.12,57.375]]])
                img = np.clip(((img*std)+mean)/255,0,1)

                extent = np.int(0), np.int(72), np.int(0), np.int(96)

                '''
                for j_idx in idx:
                    plt.figure(1, figsize=(12,8))
                    plt.subplot(5,5,j_idx+1)
                    plt.title('gt '+joints[j_idx][0])
                    plt.imshow(img, interpolation='none', extent=extent)
                    plt.imshow(target[bt, j_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none', extent=extent)
                    plt.colorbar()

                    plt.figure(2, figsize=(12,8))
                    plt.subplot(5,5,j_idx+1)
                    plt.title('pred '+joints[j_idx][0])
                    plt.imshow(img, interpolation='none', extent=extent)
                    plt.imshow(predictions[bt, j_idx], cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none', extent=extent)
                    plt.colorbar()
                '''
                plt.figure(1, figsize=(12,8))
                plt.subplot(1,3,1)
                plt.imshow(img, extent=extent)

                plt.subplot(1,3,2)
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(np.max(target[bt], axis=0), cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none', extent=extent)
                #plt.colorbar()

                plt.subplot(1,3,3)
                plt.imshow(img, interpolation='none', extent=extent)
                plt.imshow(np.max(predictions[bt], axis=0), cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none', extent=extent)

                #Display plot
                plt.show()

                #Save as .png instead of displaying plot
                #os.makedirs('./eval_vis_outputs', exist_ok=True)
                #plt.savefig('./eval_vis_outputs/'+str(data['obj_ids'][b].item())+'.png')
                #plt.close()
                
        pred, _ = self.get_max_preds(predictions)
        target, _ = self.get_max_preds(target)
        h = predictions.shape[2]
        w = predictions.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros(len(idx)) 
        avg_acc = 0 
        cnt = 0 

        for i in range(len(idx)):
            acc[i], num_cal = self.dist_acc(dists[idx[i]], self.threshold)

            avg_acc = avg_acc + acc[i]
            cnt += num_cal 

        self.correct += avg_acc
        self.total   += cnt 

        if cnt == 0:
            return torch.tensor(0)
        else:
            return self.correct/self.total

class Contrastive_Accuracy(object):
    """
    Standard accuracy computation. # of correct cases/# of total cases

    """
    def __init__(self, *args, **kwargs):
        self.margin = kwargs['cont_acc_margin']

        self.ndata = kwargs['ndata']
        self.count = 0
        self.correct = 0.
        self.total   = 0. 

    def get_accuracy(self, predictions, data):
        """
        Args:
            predictions (tuple):
                - output1 (Tensor, shape [N, D])
                - output2 (Tensor, shape [N, D])
            data        (dictionary):
                - pair_label (Tensor, shape [N,1]) 

        Return:
            Accuracy # of correct case/ # of total cases
        """

        output1, output2 = predictions
        targets = data['pair_label']

        if self.count >= self.ndata:
            self.count   = 0
            self.correct = 0
            self.total   = 0

        output1 = output1.detach().cpu()
        output2 = output2.detach().cpu()
        targets = targets.numpy()

        dist_sq = torch.sum(pow(output2 - output1, 2), 1)
        dist = torch.sqrt(dist_sq).numpy()

        self.correct += np.sum((dist < self.margin) == targets)
        self.total   += output1.shape[0]

        self.count += output1.shape[0]

        return self.correct/self.total

class Save_Video_Keypoints():
    """
    Write predictions to JSON, visualize and save as video 
    """
    def __init__(self, *args, **kwargs):
        #self.result_dir = kwargs['result_dir']

        self.load_type  = kwargs['load_type']
        self.batch_size = kwargs['batch_size'] 
        self.ndata      = kwargs['ndata']
        self.count = 0

        self.json_anns = {}
        self.output_dir = os.path.join('./outputs',kwargs['model']+'-'+kwargs['exp'])
        os.makedirs(self.output_dir, exist_ok=True)

        self.viz = kwargs['viz']

        self.eval_object = Eval_PoseTrack18_det(*args, **kwargs)
        self.conf_threshold = kwargs['conf_threshold'] 

        self.vout = cv2.VideoWriter()
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.fps = 10 #29.97
        self.vout_path  = None 
        self.prev_f_path = None 
        self.prev_keypoints = []
        self.prev_bbox = []
        self.img = None 

        self.prev_seq = None 
        self.dex = []
        self.xy_positions = {} 

        self.logger = kwargs['logger'] #wandb logging tool
        self.debug  = kwargs['debug']
        
    def get_accuracy(self, predictions, data):
        """
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

            #Open-loop, labels may or may not exist

        Return:
            0
        """

        bbox          = data['bbox']
        frame_path    = data['frame_path']
        frame_size    = data['frame_size']
        input_crop    = data.get('input_crop', data['bbox']) 
        vid_id        = data['vid_id']
        neighbor_link = data['neighbor_link']
        link_colors   = data['link_colors']
        labels        = data.get('labels', None)

        #predictions = predictions[-1].unsqueeze(1).cpu().numpy()
        predictions = predictions.cpu().numpy()
        input_crop   = input_crop.int().numpy()
        B,T,D,H,W = predictions.shape

        padding       = data.get('padding', torch.zeros(B,T,4))

        #Reshape, temporal dimension now represents multiple objects per image
        predictions  = np.reshape(predictions, (B*T,D,H,W))
        input_crop   = np.reshape(input_crop, (B,T,-1))
        idx = list(range(predictions.shape[1]))
        
        pred, maxvals = self.eval_object.get_max_preds(predictions)

        #NOTE: ONLY for PoseTrack18 dataset
        #maxvals[:,3:5] = 0 #Left-Right ears are un-annotated

        pred_mask = maxvals > self.conf_threshold
        scores  = np.clip(maxvals,0,1)
        links = neighbor_link[0]
        link_color = link_colors[0]

        pred = np.reshape(pred, (B,T,D,2))
        pred_mask = np.reshape(pred_mask, (B,T,D,1))

        for b in range(B):
            f_path   = frame_path[0][b]
            frame_w  = frame_size[0][b]
            frame_h  = frame_size[1][b]

            #Use same numpy image for all objects on same frame
            if self.prev_f_path == None:
                #'data' here must be original image
                self.img = data['data'][b].cpu().numpy()
            elif self.prev_f_path != f_path: #New frame
                frame_id = self.prev_f_path.split('/')[-1].split('.')[0]
                frame_id = int(''.join(c for c in frame_id if c.isdigit())) #strip non-numbers

                anns = self.eval_object.assign_ids(self.json_anns, self.prev_seq, frame_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.img, label=self.dex)
                self.prev_keypoints = []
                self.prev_bbox = []
                self.dex       = [] 
                max_tid = 0 

                #Draw assigned track ids on image
                font = cv2.FONT_HERSHEY_SIMPLEX 
                if not anns is None:
                    for ann in anns:
                        tid  = ann['track_id']
                        kpts = np.array(ann['keypoints']).reshape(D,3)
                        mask = kpts[:,2] > 0

                        xmin, ymin, _ = np.min(kpts[mask],0)

                        #cv2.putText(self.img, str(tid)+', '+ann['label'], (int(xmin),int(ymin)), font, 1.75, (0,255,255), 2, cv2.LINE_AA)

                        max_tid = max(tid, max_tid)
                        '''
                        #Save centroid positions
                        det_bbox = ann['det_box']
                        x_avg = (det_bbox[2]+det_bbox[0])/2
                        y_avg = (det_bbox[3]+det_bbox[1])/2
                        pos = ','.join((str(x_avg), str(y_avg), ann['label']))
                        if frame_id not in self.xy_positions:
                            self.xy_positions[frame_id] = {tid:pos}
                        else:
                            self.xy_positions[frame_id][tid] = pos 
                        '''
                
                self.vout.write(self.img)

                #'data' here must be original image
                self.img = data['data'][b].cpu().numpy()

            #seq_name = f_path.split('/')[-2]
            seq_name = vid_id[b]
            if self.prev_seq != seq_name:
                '''
                if len(self.xy_positions.values()) > 0:
                    import matplotlib as mpl
                    mpl.use('Agg')
                    import matplotlib.pyplot as plt 

                    tids    = {}
                    for xy in self.xy_positions.values():
                        for tid, pos in xy.items():
                            if tid not in tids: tids[tid] = []

                            #pos = xy.get(tid, '-1,-1,n/a')
                            tids[tid].append(pos)

                    plt.ylim(top=0, bottom=frame_h)
                    plt.xlim(left=0,right=frame_w)
                    for tid, pos in tids.items():
                        x   = np.array([float(dat.split(',')[0]) for dat in pos])
                        y   = np.array([float(dat.split(',')[1]) for dat in pos])
                        dex = [dat.split(',')[2] for dat in pos if dat.split(',')[2] != 'n/a']

                        dex = max(set(dex), key=dex.count)
                        #color = list(np.random.choice(range(256), size=3))
                        x = x[x != -1]
                        y = y[y != -1]
                        plt.plot(x, y)
                        plt.plot(x[0],y[0],'go')
                        plt.plot(x[-1],y[-1],'ro')
                        plt.text(np.mean(x), np.mean(y), str(tid)+','+dex, fontsize=15)
                    
                    #plt.show()
                    plt.savefig(os.path.join(self.output_dir, self.prev_seq+'.png'))
                    plt.close()
                '''

                self.prev_seq = seq_name 
                self.vout.release()

                if not self.debug and self.vout.isOpened():
                    self.logger.log({seq_name:wandb.Video(self.vout_path, caption='Output video example', fps=self.fps, format='mp4')})
        
                self.vout_path = os.path.join(self.output_dir,seq_name+'.mp4')
                print(self.vout_path)
                self.vout.open(self.vout_path, self.fourcc, self.fps, (frame_w.item(), frame_h.item()), True) 

                self.json_anns[seq_name] = {'annotations':[]}

                self.xy_positions = {}

            for t in range(T):
                crop     = input_crop[b,t]
                pad      = padding[b,t] #pl, pt, pr, pb
                if crop[0] == -1:
                    continue

                keypoints = np.concatenate((pred[b,t], pred_mask[b,t]), axis=-1)

                if 'stride' in data:
                    #Scale coordinates from heatmap size
                    sw, sh = data['stride'][b]
                    keypoints[:,0] *= sw.item()
                    keypoints[:,1] *= sh.item()

                    #Apply inverse transform w/ respect original image
                    inv_trans = data['inv_trans']
                    keypoints[:,:2] = self.eval_object.transform_pts(keypoints[:,:2], inv_trans[b,t])
                    keypoints[keypoints[:,2] < 1] = 0
                else:
                    #scale coordinates to crop size
                    crop_h = (crop[3]-crop[1]).item()
                    crop_w = (crop[2]-crop[0]).item()
                    keypoints[:,0] *= (crop_w/W)
                    keypoints[:,1] *= (crop_h/H)

                    #Undo crop
                    keypoints[:,0] += crop[0].item()
                    keypoints[:,1] += crop[1].item()

                    #Subtract padding if was added
                    keypoints[:,0] -= pad[0].item()
                    keypoints[:,1] -= pad[1].item()

                if np.sum(keypoints[...,-1]>self.conf_threshold) >= 1:
                    self.prev_keypoints.append(keypoints)
                    self.prev_bbox.append(bbox[b,t].float())

                    #dexter   = 'right' if labels[b].item() else 'left'
                    #self.dex.append(dexter)

                box_line_size = 1 + np.floor(frame_w.item()/400).astype('int')
                line_size = 4 * max(1, np.floor(frame_w.item()/400).astype('int'))
                rad_size  = 3 * max(1, np.floor(frame_w.item()/400).astype('int'))
                xmin, ymin, xmax, ymax = crop #cropped input to model

                #cv2.rectangle(self.img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), color=(0,0,255)) #draw model input

                xmin, ymin, xmax, ymax = bbox[b,t].int()
                cv2.rectangle(self.img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), color=(255,255,255), thickness=box_line_size) #draw tight bbox

                for idx,(p1,p2) in enumerate(links):
                    x1, y1, c1 = keypoints[p1]
                    x2, y2, c2 = keypoints[p2]

                    #Filter keypoints outside of tight bounding box
                    if x1 < xmin or x1 > xmax or y1 < ymin or y1 > ymax:
                        c1 = 0
                    if x2 < xmin or x2 > xmax or y2 < ymin or y2 > ymax:
                        c2 = 0

                    col = col1 = col2 = link_color[idx].tolist() #R,G,B
                    r1 = r2 = rad_size 

                    #let's make the wrist connection more obvious
                    if p1 == 0:
                        col1 = (255,255,255)
                        r1 = np.floor(rad_size*1.5).astype('int')
                    if p2 == 0:
                        col2 = (255,255,255)
                        r2 = np.floor(rad_size*1.5).astype('int')

                    if c1 != 0 and c2 != 0:
                        cv2.line(self.img, (int(x1),int(y1)), (int(x2),int(y2)), (col[2],col[1],col[0]), line_size)
                    if c1 != 0:
                        cv2.circle(self.img, (int(x1),int(y1)), radius=r1, color=(col1[2],col1[1],col1[0]), thickness=-1)
                    if c2 != 0 :
                        cv2.circle(self.img, (int(x2),int(y2)), radius=r2, color=(col2[2],col2[1],col2[0]), thickness=-1)

            self.prev_f_path = f_path 

        self.count += B

        if self.count >= self.ndata:
            frame_id = self.prev_f_path.split('/')[-1].split('.')[0]
            frame_id = int(''.join(c for c in frame_id if c.isdigit())) #strip non-numbers

            anns = self.eval_object.assign_ids(self.json_anns, self.prev_seq, frame_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.img, label=self.dex)
            self.prev_keypoints = []
            self.prev_bbox = []

            #Draw assigned track ids on image
            font = cv2.FONT_HERSHEY_SIMPLEX 
            if not anns is None:
                for ann in anns:
                    tid  = ann['track_id']
                    kpts = np.array(ann['keypoints']).reshape(D,3)
                    mask = kpts[:,2] > 0

                    xmin, ymin, _ = np.min(kpts[mask],0)

                    #cv2.putText(self.img, str(tid)+', '+ann['label'], (int(xmin),int(ymin)), font, 1.75, (0,255,255), 2, cv2.LINE_AA)

                    '''                    
                    #Save centroid positions
                    det_bbox = ann['det_box']
                    x_avg = (det_bbox[2]+det_bbox[0])/2
                    y_avg = (det_bbox[3]+det_bbox[1])/2
                    pos = ','.join((str(x_avg), str(y_avg), ann['label']))
                    if frame_id not in self.xy_positions:
                        self.xy_positions[frame_id] = {tid:pos}
                    else:
                        self.xy_positions[frame_id][tid] = pos 
                    '''

            self.vout.write(self.img)

            '''
            import matplotlib.pyplot as plt 
            tids = {} 
            for xy in self.xy_positions.values():
                for tid in range(0,max_tid+1):
                    if tid not in tids: tids[tid] = []

                    pos = xy.get(tid, '-1,-1,n/a')
                    tids[tid].append(pos)

            plt.ylim(top=0, bottom=frame_h)
            plt.xlim(left=0,right=frame_w)
            for tid, pos in tids.items():
                x   = np.array([float(dat.split(',')[0]) for dat in pos])
                y   = np.array([float(dat.split(',')[1]) for dat in pos])
                dex = [dat.split(',')[2] for dat in pos if dat.split(',')[2] != 'n/a']

                dex = max(set(dex), key=dex.count)
                #color = list(np.random.choice(range(256), size=3))
                x = x[x != -1]
                y = y[y != -1]
                plt.plot(x, y)
                plt.text(np.mean(x), np.mean(y), str(tid)+','+dex, fontsize=15)
            
            #plt.show()
            plt.savefig(os.path.join(self.output_dir, seq_name+'.png'))
            plt.close()
            '''

        return 0

class Save_Frame_Video_Heatmaps():
    """
    Reproject all heatmap predictions to full frame videos
    """
    def __init__(self, *args, **kwargs):
        #self.result_dir = kwargs['result_dir']

        self.load_type  = kwargs['load_type']
        self.batch_size = kwargs['batch_size'] 
        self.ndata      = kwargs['ndata']
        self.count = 0

        self.json_anns = {}
        self.output_dir = os.path.join('./outputs',kwargs['model']+'-'+kwargs['exp'])
        os.makedirs(self.output_dir, exist_ok=True)

        self.viz = kwargs['viz']

        self.eval_object = Eval_PoseTrack18_det(*args, **kwargs)
        self.conf_threshold = kwargs['conf_threshold'] 

        self.vout = cv2.VideoWriter()
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #self.fps = 29.97
        self.fps = 10
        self.vout_path  = None 
        self.prev_f_path = None 
        self.img = None 
        self.rgb_img = None
        self.img_hm = None 

        self.save_feat = True #Save each heatmap image as a feature
        self.save_feat_dir = os.path.join(kwargs['save_feat_dir'], kwargs['model']+'-'+kwargs['exp'])

        self.prev_seq = None 

        self.logger = kwargs['logger'] #wandb logging tool
        self.debug  = kwargs['debug']
        
    def get_accuracy(self, predictions, data):
        """
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

            #Open-loop, labels may or may not exist

        Return:
            0
        """

        bbox          = data['bbox']
        frame_path    = data['frame_path']
        frame_size    = data['frame_size']
        input_crop    = data.get('input_crop', data['bbox']) 
        vid_id        = data['vid_id']

        predictions = predictions.cpu().numpy()
        input_crop   = input_crop.int().numpy()
        B,T,D,H,W = predictions.shape

        padding       = data.get('padding', torch.zeros(B,T,4))

        #Reshape, temporal dimension now represents multiple objects per image
        predictions  = np.reshape(predictions, (B*T,D,H,W))
        input_crop   = np.reshape(input_crop, (B,T,-1))
        idx = list(range(predictions.shape[1]))
        
        #NOTE: ONLY for PoseTrack18 dataset
        #maxvals[:,3:5] = 0 #Left-Right ears are un-annotated

        for b in range(B):
            f_path   = frame_path[0][b]
            frame_w  = frame_size[0][b]
            frame_h  = frame_size[1][b]
            seq_name = vid_id[b]

            #Use same numpy image for all objects on same frame
            if self.prev_f_path == None:
                #'data' here must be original image
                if self.viz:
                    self.rgb_img = data['data'][b].cpu().numpy()
                self.img = np.zeros((frame_h, frame_w, 3), dtype=np.float32)
                self.img_hm = np.zeros((frame_h, frame_w, D), dtype=np.float32)
            elif self.prev_f_path != f_path: #New frame
                frame_id = self.prev_f_path.split('/')[-1].split('.')[0]
                frame_id = int(''.join(c for c in frame_id if c.isdigit())) #strip non-numbers

                #Normalize and quantize
                out_img = self.img / np.max(self.img)
                out_img = np.array(out_img * 255, dtype=np.uint8)
                #self.vout.write(out_img)

                if self.save_feat:
                    f_name = self.prev_f_path.split('/')[-1].split('.')[0]
                    feat_dir = os.path.join(self.save_feat_dir, self.prev_f_path.split('/')[-2])
                    os.makedirs(feat_dir, exist_ok=True)

                    save_path = os.path.join(feat_dir, f_name+'.npy')
                    np.save(save_path, self.img_hm)

                if self.viz:
                    import matplotlib.pyplot as plt 
                    plt.imshow(self.rgb_img)
                    plt.imshow(np.max(self.img_hm,axis=-1), cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none')
                    plt.colorbar()
                    plt.show() 

                # new image
                self.img = np.zeros((frame_h, frame_w, 3), dtype=np.float32)
                self.img_hm = np.zeros((frame_h, frame_w, D), dtype=np.float32)

            #seq_name = f_path.split('/')[-2]
            if self.prev_seq != seq_name:
                self.prev_seq = seq_name 

                self.vout.release()

                if not self.debug and self.vout.isOpened():
                    self.logger.log({seq_name:wandb.Video(self.vout_path, caption='Output video example', fps=self.fps, format='mp4')})
        
                self.vout_path = os.path.join(self.output_dir,seq_name+'.mp4')
                print(self.vout_path)
                self.vout.open(self.vout_path, self.fourcc, self.fps, (frame_w, frame_h), True) 

                self.json_anns[seq_name] = {'annotations':[]}

            for t in range(T):
                crop     = input_crop[b,t]
                pad      = padding[b,t] #pl, pt, pr, pb
                if crop[0] == -1:
                    continue

                if 'stride' in data:
                    #Scale coordinates from heatmap size
                    sw, sh = data['stride'][b]
                    keypoints[:,0] *= sw.item()
                    keypoints[:,1] *= sh.item()

                    #Apply inverse transform w/ respect original image
                    inv_trans = data['inv_trans']
                    keypoints[:,:2] = self.eval_object.transform_pts(keypoints[:,:2], inv_trans[b,t])
                    keypoints[keypoints[:,2] < 1] = 0
                else:
                    #scale coordinates to crop size
                    crop_h = (crop[3]-crop[1]).item()
                    crop_w = (crop[2]-crop[0]).item()

                    pl,pt,pr,pb = pad 
                    #adjust crop to visible area by undoing padding
                    x1,y1,x2,y2 = crop
                    x1 += pl.item() 
                    y1 += pt.item() 
                    x2 -= pr.item()
                    y2 -= pb.item() 

                    temp_ = F.interpolate(torch.from_numpy(predictions[b,None]), size=(crop_h,crop_w))

                    pb = crop_h if not pb else -pb #check if non-zero, and adjust for array slicing
                    pr = crop_w if not pr else -pr

                    temp_    = temp_[0][:,pt:pb,pl:pr]
                    temp_img = torch.max(temp_,dim=0)[0].numpy()
                    self.img[y1-pt.item():y2-pt.item(),x1-pl.item():x2-pl.item()] += temp_img[:,:,None] 
                    self.img_hm[y1-pt.item():y2-pt.item(),x1-pl.item():x2-pl.item(),:] += temp_.permute(1,2,0).numpy()

            self.prev_f_path = f_path 

        self.count += B

        if self.count >= self.ndata:
            frame_id = self.prev_f_path.split('/')[-1].split('.')[0]
            frame_id = int(''.join(c for c in frame_id if c.isdigit())) #strip non-numbers

            if self.save_feat:
                f_name = self.prev_f_path.split('/')[-1].split('.')[0]
                feat_dir = os.path.join(self.save_feat_dir, seq_name)
                os.makedirs(feat_dir, exist_ok=True)

                save_path = os.path.join(feat_dir, f_name+'.npy')
                np.save(save_path, self.img_hm)
            
            if self.viz:
                import matplotlib.pyplot as plt 
                plt.imshow(self.rgb_img)
                plt.imshow(np.max(self.img_hm,axis=-1), cmap='jet', alpha=0.5, vmin=0, vmax=1, interpolation='none')
                plt.colorbar()
                plt.show() 

            #self.vout.write(self.img[...,::-1])
            self.vout.write(self.img)

        return 0

class Eval_PoseTrack18_det():
    """
    Write predictions for submission in JSON format. For PoseTrack18 submission.
    From bounding box detections, not ground truth
    """
    def __init__(self, *args, **kwargs):
        #self.result_dir = kwargs['result_dir']

        self.load_type  = kwargs['load_type']
        self.batch_size = kwargs['batch_size'] 
        self.ndata      = kwargs['ndata']
        self.count = 0

        self.correct = 0
        self.total   = 0

        self.json_anns = {}

	    #Any keypoints below this confidence threshold will be zero-ed out
        self.conf_threshold = kwargs['conf_threshold'] 

        #Run PoseTrack evaluation directly from here
        poseval_dir = kwargs.get('poseval_dir', '/z/home/natlouis/poseval_hand/py')
        #os.environ['PYTHONPATH'] = os.path.join(poseval_dir,'..','py-motmetrics:'+os.getenv('PYTHONPATH',''))
        self.exec_loc = os.path.join(poseval_dir, 'evaluate.py')

        #self.exec_loc = 'poseval.evaluate' #using faster poseval, run as a module
        run_id = kwargs.get('run_id', kwargs['exp'])
        self.pred_dir = os.path.join(poseval_dir, 'prediction-'+kwargs['model']+'-'+run_id+'/')

        #TODO: Cheap and easy fix, shouldn't leave permanently like this
        tags = kwargs['tags']
        if tags:
            fold = tags[0]
            self.targ_dir = os.path.join(poseval_dir, 'temp_target_'+fold+'/')
        else:
            self.targ_dir = os.path.join(poseval_dir, 'target/')
        
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.targ_dir, exist_ok=True)
        self.match_strategy = kwargs['match_strategy']
        self.l2_margin      = kwargs['l2_margin']
        self.spa_con_thresh = kwargs['spa_con_thresh']
        self.last_T_frames  = kwargs['last_T_frames']

        if self.match_strategy == 'gcn':
            self.gcn_margin  = kwargs['cont_acc_margin']
            gcn_model_ckpt   = kwargs['gcn_checkpoint']
            self.gcn_model   = GCN(in_channels=kwargs['in_channels'], edge_importance_weighting=True,\
                                    layout='hand', partition_strategy='spatial', gcn_vis_embed=kwargs['gcn_vis_embed']).to('cuda')
            self.gcn_model.load_state_dict(torch.load(gcn_model_ckpt)['state_dict'])
            self.gcn_model.eval()

        self.prev_vis_feat   = [] #saved video features for each object
        self.last_vis_feat   = {} #last tracked vis_features with each assigned track id

        self.prev_vid_id    = 'null'
        self.prev_image_id  = 'null'
        self.prev_frame     = None 
        self.prev_seq_name  = 'null'
        self.prev_keypoints = [] #saved keypoints from previous image
        self.prev_bbox      = [] #saved bbox detections for each object
        self.hungarian = Hungarian()

        self.flow_data_root = kwargs.get('flow_data_root', None)

        self.viz = kwargs['viz']

        self.logger = kwargs['logger'] #wandb logging tool
        self.debug  = kwargs['debug']

    def get_max_preds(self, batch_heatmaps):
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

    def transform_pts(self, pt, t):
        '''
        Apply transform to predicted coordinates, w/ respect to original image
        '''
        new_pt = np.hstack((pt, np.ones((pt.shape[0],1))))
        new_pt = np.dot(t, new_pt.transpose()).transpose()

        return new_pt
    
    def keypoints_to_bbox(self, kpts, sc, f_size):
        '''
        Capture largest bounding box around predicted keypoints. Using for Spatial Consistency i.e. IoU 
            kpts (Tensor, shape [N,J,3] or [J,3])
            sc   (Tensor, shape []) : scale width and height
            f_size (Tuple, shape (2)): (width, height)
        '''
        assert sc > 0
        max_width  = f_size[0]*1.0
        max_height = f_size[1]*1.0

        bboxes = []

        if kpts.dim() < 3:
            kpts = kpts.unsqueeze(0)
        
        for kpt in kpts:
            mask = kpt[...,-1] > self.conf_threshold 

            xmin = torch.min(kpt[mask,0])
            ymin = torch.min(kpt[mask,1])
            xmax = torch.max(kpt[mask,0])
            ymax = torch.max(kpt[mask,1])

            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w * 0.5
            cy = ymin + h * 0.5

            if w == 0 or h == 0: #invalid bboxes will occupy entire screen, should induce low IoU
                w = max_width
                h = max_height
                
            #Enlarge by scale
            w *= sc
            h *= sc

            xmin = torch.clamp(cx - (0.5 * w), min=0)
            ymin = torch.clamp(cy - (0.5 * h), min=0)
            xmax = torch.clamp(cx + (0.5 * w), max=max_width)
            ymax = torch.clamp(cy + (0.5 * h), max=max_height)

            bboxes.append(torch.tensor([xmin, ymin, xmax, ymax]))

        return torch.stack(bboxes)

    def assign_ids(self, json_anns, seq_name, image_id, keypoints, det_bboxes, frame_dims, frame, viz=False, vis_feats=None, label=None):
        if len(keypoints) == 0:
            return #No keypoints to track

        frame_w, frame_h = frame_dims 
        #keypoints: list of Nx3 arrays (x,y,score)
        anns = []
        tids = {}

        kpts_curr_frame = torch.tensor(np.array(keypoints))
        bbox_curr_frame = torch.stack(det_bboxes)
        prev_anns = json_anns[seq_name]['annotations']

        if vis_feats:
            vis_feats_curr_frame = torch.stack(vis_feats)

        #Compare to poses from last T frames 
        T = self.last_T_frames
     
        if len(prev_anns) > 0:#Previous frame annotations exist
            tids_prev_frame = []
            kpts_prev_frame = []
            bbox_prev_frame = []

            uniq_frames = []
            for ann in reversed(prev_anns):
                uniq_frames.append(ann['image_id'])
                uniq_frames = list(set(uniq_frames))

                if len(uniq_frames) > T:
                    break

                if ann['image_id'] < image_id: #previous frame(s)
                    tids_prev_frame.append(ann['track_id'])
                    kpts_prev_frame.append(torch.tensor(ann['keypoints']).view(-1,3))
                    bbox_prev_frame.append(torch.tensor(ann['det_box']))

            #assign ids to previous T frames
            kpts_prev_frame = torch.stack(kpts_prev_frame)
            bbox_t          = torch.stack(bbox_prev_frame)

            if self.flow_data_root is not None:
                #Use flow-based similarity
                #Instead of comparing to the exact bounding box (or keypoints) from the previous frame,
                #let's use optical to propogate the bounding box (or keypoints) to the current frame

                #Get (str) frame num
                frame_num = str(image_id)[-4:].zfill(6)
                #Load optical flow image from frame id
                opt_flow_file = os.path.join(self.flow_data_root, seq_name, frame_num+'.flo')
                if os.path.isfile(opt_flow_file):
                    flow_data = fl.read_flow(os.path.join(self.flow_data_root, seq_name, frame_num+'.flo'))

                    #zero pad flow_data to match image dimensions (cropped in FlowNet2C)
                    pad_h = frame_h - flow_data.shape[0]
                    pad_w = frame_w  - flow_data.shape[1]

                    top = bottom = left = right = 0
                    if pad_h > 0:
                        top = bottom = int(pad_h/2)
                    if pad_w > 0:
                        left = right = int(pad_w/2)
                    flow_data = np.pad(flow_data, ((top,bottom), (left,right),(0,0)), 'constant')

                    #Add flow offsets to previous bbox and previous keypoints
                    _kpts_prev_frame = kpts_prev_frame.int()
                    _bbox_t          = bbox_t.int()

                    for idx, (kpt, bbox) in enumerate(zip(_kpts_prev_frame, _bbox_t)):
                        x1,y1,x2,y2 = bbox
                        _x1, _y1 = flow_data[max(0, min(int(y1), frame_h-1)), max(0, min(int(x1), frame_w-1))]
                        _x2, _y2 = flow_data[max(0, min(int(y2), frame_h-1)), max(0, min(int(x2), frame_w-1))]

                        bbox_t[idx] += torch.tensor([_x1,_y1,_x2,_y2])

                        flow = flow_data[np.clip(kpt[:,1], 0, frame_h-1), np.clip(kpt[:,0], 0, frame_w-1)] #y,x
                        kpts_prev_frame[idx,:,0] += torch.tensor(flow[:,1])
                        kpts_prev_frame[idx,:,1] += torch.tensor(flow[:,0])

                else:
                    print('{} does not exist'.format(opt_flow_file))
            try:
                if self.match_strategy == 'l2': #L2-Distance between points
                    margin = self.l2_margin

                    kpts_curr_frame = torch.tensor(keypoints)
                    M = kpts_curr_frame.shape[0]
                    N = kpts_prev_frame.shape[0]
                    costs = 1e10 * np.ones((M,N))
                    for i, kp in enumerate(torch.tensor(keypoints)):
                        dists = torch.mean(torch.sqrt((kpts_prev_frame[...,:2] - kp[:,:2]).pow(2).sum(2)),dim=1)
                        d = dists.shape[0]
                        costs[i,:d] = dists

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        dist = costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if dist < margin and _tid not in tids.values():
                            tids[tcurr] = _tid
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

                elif self.match_strategy == 'sc': #Spatial Consistency
                    threshold = self.spa_con_thresh

                    M = bbox_curr_frame.shape[0]
                    N = bbox_t.shape[0]
                    costs = 1e10 * np.ones((M,N))
                    #Get IOU between all pairs
                    for i, box_p in enumerate(bbox_curr_frame):
                        intersect_area = IOU().intersect(box_p, bbox_t) 

                        box_p_area = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
                        box_t_area = (bbox_t[:,2] - bbox_t[:,0]) * (bbox_t[:,3] - bbox_t[:,1])
                        union = box_p_area + box_t_area - intersect_area 

                        union = torch.clamp(union, min=0.001)
                        overlaps = intersect_area/union

                        d = overlaps.shape[0]
                        costs[i,:d] = 1 - overlaps

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        overlap = 1 - costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if overlap > threshold and _tid not in tids.values():
                            tids[tcurr] = _tid 
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

                elif self.match_strategy == 'gcn': #Graph Convolutional Network
                    margin = self.gcn_margin 

                    M = kpts_curr_frame.shape[0]
                    N = kpts_prev_frame.shape[0]
                    costs = 1e10 * np.ones((M,N), dtype=np.int32)

                    #Normalize keypoints to [0,1] w/ respect to detection box size
                    widths = bbox_curr_frame[:,2]-bbox_curr_frame[:,0]
                    heights= bbox_curr_frame[:,3]-bbox_curr_frame[:,1]
                    norm_kpts_curr = kpts_curr_frame - torch.cat((bbox_curr_frame[:,:2], torch.zeros(M,1)), dim=1).unsqueeze(1)
                    norm_kpts_curr /= torch.stack((widths, heights, torch.ones(M)),dim=1).unsqueeze(1)

                    widths = bbox_t[:,2]-bbox_t[:,0]
                    heights= bbox_t[:,3]-bbox_t[:,1]
                    norm_kpts_prev = kpts_prev_frame - torch.cat((bbox_t[:,:2], torch.zeros(N,1)), dim=1).unsqueeze(1)
                    norm_kpts_prev /= torch.stack((widths, heights, torch.ones(N)),dim=1).unsqueeze(1)

                    #Zero-out unannotated points
                    norm_kpts_curr[norm_kpts_curr[...,2] == 0] = -1
                    norm_kpts_prev[norm_kpts_prev[...,2] == 0] = -1

                    if vis_feats:
                        #Get corresponding visual features
                        vis_feats_last_frame = torch.zeros((N,vis_feats_curr_frame.shape[-1]))
                        for i,_tid in enumerate(tids_prev_frame):
                            vis_feats_last_frame[i] = self.last_vis_feat[_tid]

                        #Extract joint graph embedding for all keypoints + vis features
                        with torch.no_grad():
                            curr_embeds  = self.gcn_model.extract_joint_embed(vis_feats_curr_frame.to('cuda'), norm_kpts_curr[:,None,:,:2].to('cuda'))
                            prev_embeds  = self.gcn_model.extract_joint_embed(vis_feats_last_frame.to('cuda'), norm_kpts_prev[:,None,:,:2].to('cuda'))
                    else:
                        #Extract graph embedding for all keypoints
                        with torch.no_grad():
                            curr_embeds  = self.gcn_model.extract_feature(norm_kpts_curr[:,None,:,:2].to('cuda'))
                            prev_embeds  = self.gcn_model.extract_feature(norm_kpts_prev[:,None,:,:2].to('cuda'))

                    #Compute distance between all pairs
                    for i, ce in enumerate(curr_embeds):
                        dists = torch.sqrt((prev_embeds - ce).pow(2).sum(-1))
                        d = dists.shape[0]
                        costs[i,:d] = dists.cpu()

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        dist = costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if dist < margin and _tid not in tids.values():
                            tids[tcurr] = _tid
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)
            except BaseException as e:
                print('Error: {}'.format(e))
                print('image_id: {}'.format(image_id))
                import pdb; pdb.set_trace()

            #If many new objects are introduced, they will not get assigned from matching algorithm
            #Because there won't be a previous id to assign to. The costs matrix can be wide, but not tall
            #Happens a lot with cut-scenes
            if M > len(results):
                for tcurr in range(M):
                    if tcurr in tids: 
                        continue 

                    #print('New object introduced at frame: {}'.format(image_id))
                    tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

        else: #First frame, initialize all track ids
            for i in range(len(keypoints)):
                tids[i] = i

        assert(len(list(set(tids.values()))) == len(keypoints))

        self.last_vis_feat = {}
        for i,(kp,det_box) in enumerate(zip(keypoints, det_bboxes)):
            anns.append({'image_id':image_id,
                   #'track_id': 0, #NOTE: As a test, give all the track ids a value of zero
                   'track_id': tids[i],
                   'keypoints':kp.reshape(-1).tolist(),
                   'det_box':det_box.reshape(-1).tolist(),
                   #'label':label[i],
                   'scores':kp[:,-1].tolist()})

            if vis_feats:
                self.last_vis_feat[tids[i]] = vis_feats_curr_frame[i]  

        if viz:
            #must use original image
            img = frame.numpy()

            for i, bbox in enumerate(bbox_curr_frame.int()):
                xmin, ymin, xmax, ymax = bbox 

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color=(255,255,255), thickness=2)

                for idx,(p1,p2) in enumerate(self.links):
                    x1, y1, c1 = kpts_curr_frame[i, p1]
                    x2, y2, c2 = kpts_curr_frame[i, p2]

                    c = self.link_color[idx].tolist() #R,G,B
                    if c1 != 0:
                        cv2.circle(img, (x1,y1), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                    if c2 != 0 :
                        cv2.circle(img, (x2,y2), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                    if c1 != 0 and c2 != 0:
                        cv2.line(img, (x1,y1), (x2,y2), (c[2],c[1],c[0]), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'track id:'+str(tids[i]), (xmin, ymin), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

            #cv2.imshow('Output', img)
            cv2.imshow('Output', img[...,::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        json_anns[seq_name]['annotations'].extend(anns)

        return anns

    def get_accuracy(self, predictions, data):
        """
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

        Return:
            Accuracy # of correct case/ # of total cases
        """

        bbox         = data['bbox']
        #bbox         = data['input_crop']

        frame_path   = data['frame_path']
        frame_id     = data['frame_ids']
        nframes      = data['nframes']
        frame_size   = data['frame_size']
        vid_ids      = data['vid_id']

        neighbor_link = data['neighbor_link']
        link_colors   = data['link_colors']

        self.links = neighbor_link[0]
        self.link_color = link_colors[0]

        vis_feats = None
        if isinstance(predictions, tuple):
            _, _, _, _, _, out6 = predictions
            predictions = out6.unsqueeze(1)
        elif isinstance(predictions, dict):
            vis_feats   = predictions['feat']
            predictions = predictions['outputs']

        predictions = predictions.cpu().numpy()
        B,T,D,H,W = predictions.shape

        #Reshape, temporal dimension now represents multiple objects per image
        predictions = np.reshape(predictions, (B*T,D,H,W))
        idx = list(range(predictions.shape[1]))
        
        pred, maxvals = self.get_max_preds(predictions)
        #maxvals[:,3:5] = 0 #Left-Right ears are un-annotated
        pred_mask = maxvals > 0
        scores  = np.clip(maxvals,0,1)

        #if self.load_type == 'test' and self.batch_size == 1:
        cat = data['categories']

        categories = {}
        categories['supercategory'] = cat['supercategory'][0]
        categories['id']        = int(cat['id'][0])
        categories['name']      = cat['name'][0]
        categories['skeleton']  = cat['skeleton'][0].long().numpy().tolist()
        categories['keypoints'] = [k[0] for k in cat['keypoints']]

        for bt in range(B*T):
            b = int(bt/T)
            t = bt % T

            f_path   = frame_path[0][b]
            frame_w  = frame_size[0][b]
            frame_h  = frame_size[1][b]
            image_id = frame_id[b,0].item()
            vid_id   = vid_ids[b]

            if f_path != 'None':
                seq_name = vid_ids[b]

            if bbox[b,t,0] == -1: #skipped detection (low score) or no detection
                is_labeled = False
                ann = None
                
                if t >= len(frame_path): #no frames at time t
                    continue
            else:
                is_labeled = True

                keypoints = np.concatenate((pred[bt], pred_mask[bt]), axis=-1)

                if 'stride' in data:
                    #Scale coordinates from heatmap size
                    sw, sh = data['stride'][b]
                    keypoints[:,0] *= sw.item() 
                    keypoints[:,1] *= sh.item()

                    #Apply inverse transform w/ respect original image
                    inv_trans = data['inv_trans']
                    keypoints[:,:2] = self.transform_pts(keypoints[:,:2], inv_trans[b,t])
                    keypoints[:,-1] = scores[bt].squeeze() #Add scores to keypoints
                    thresh = 0
                    keypoints[keypoints[:,2] < thresh] = 0

                else:
                    input_crop = data.get('input_crop', data['bbox'])
                    padding    = data.get('padding', torch.zeros(B,T,4))

                    crop = input_crop[b,t]
                    pad  = padding[b,t] #pl, pt, pr, pb

                    #scale coordinates to crop size
                    crop_h = (crop[3]-crop[1]).item()
                    crop_w = (crop[2]-crop[0]).item()
                    keypoints[:,0] *= (crop_w/W)
                    keypoints[:,1] *= (crop_h/H)

                    #Undo crop
                    keypoints[:,0] += crop[0].item()
                    keypoints[:,1] += crop[1].item()

                    #Subtract padding if was added
                    keypoints[:,0] -= pad[0].item()
                    keypoints[:,1] -= pad[1].item()

                    keypoints[:,-1] = scores[bt].squeeze() #Add scores to keypoints

                #if np.sum(keypoints[:,-1]) == 0: #Skip poses w/ no valid keypoints
                #    continue 
                
                temp_score = np.copy(scores[bt].reshape(-1))

                #Zero-out values below confidence threshold
                mask = temp_score < self.conf_threshold
                temp_score[mask] = 0
                keypoints[mask] = 0

                #Filter keypoints outside of tight bounding box
                xmin, ymin, xmax, ymax = bbox[b,t]
                for j in range(D):
                    x1, y1, _ = keypoints[j]
                    if x1 < xmin or x1 > xmax or y1 < ymin or y1 > ymax:
                        keypoints[j] = 0

                if 'data' in data:
                    frame = data['data'][b]
                else:
                    frame = None

                #Assigning Track Id
                if self.prev_vid_id != vid_id: #New video
                    if self.prev_seq_name != 'null':
                        self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.prev_frame, self.viz, self.prev_vis_feat)

                    self.prev_vid_id    = vid_id
                    self.prev_image_id  = image_id 
                    self.prev_seq_name  = seq_name
                    self.prev_frame     = frame 
                    self.prev_keypoints = []
                    self.prev_bbox      = []
                    self.prev_vis_feat  = []

                    json_data = {'annotations':[], 'categories':[categories]}    
                    self.json_anns[seq_name] = json_data 

                elif self.prev_image_id != image_id: #Same video, new frame
                    self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.prev_frame, self.viz, self.prev_vis_feat)
                    
                    self.prev_image_id = image_id
                    self.prev_frame     = frame 
                    self.prev_keypoints = []
                    self.prev_bbox      = []
                    self.prev_vis_feat  = []

                if np.sum(keypoints[...,-1]>self.conf_threshold) >= 1: #Only add if atleast 1 keypoint is above threshold
                    self.prev_keypoints.append(keypoints)
                    self.prev_bbox.append(bbox[b,t].float())
                    if vis_feats is not None:
                        self.prev_vis_feat.append(vis_feats[b,t])

        self.count += B

        if self.count >= self.ndata:
            #Last frame
            self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), frame, self.viz, self.prev_vis_feat)

            for seq_name, json_data in self.json_anns.items():
                #Add images from gt annotation b/c not all frames labeled and annotated 
                gt_json_file = os.path.join(self.targ_dir, seq_name+'.json')
                
                with open(gt_json_file, 'r') as f:
                    gt_json_data = json.load(f)

                images = gt_json_data['images']
                for idx in range(len(images)): #only keep keys: is_labeled, file_name, id (probably makes no difference)
                    images[idx].pop('has_no_densepose')
                    images[idx].pop('nframes')
                    images[idx].pop('vid_id')
                    images[idx].pop('frame_id')

                json_data['images'] = images 
                json_file = os.path.join(self.pred_dir, seq_name+'.json')

                with open(json_file, 'w') as f:
                    json.dump(json_data, f)

            print('Running PoseTrack Pose Estimation Eval')
            try:
                pose_est = subprocess.check_output(['python', self.exec_loc,'--groundTruth='+self.targ_dir,\
                        '--predictions='+self.pred_dir,\
                        '--evalPoseEstimation'])
            except subprocess.CalledProcessError as e:
                sys.exit(e.output)

            pose_est = pose_est.decode().split('\n')[-4:]

            metric  = pose_est[0]
            headers = pose_est[1].replace('\\','').replace(' ','').split('&')[1:]
            scores  = pose_est[2].replace('\\','').replace(' ','').split('&')[1:]

            ap_total = float(scores[-1])*.01

            print(metric)
            print(headers)
            print(scores)

            if not self.debug:
                for h,s in zip(headers, scores):
                    self.logger.log({'AP '+h:float(s)})

            print('Running PoseTrack Pose Tracking Eval')
            try:
                pose_track = subprocess.check_output(['python', self.exec_loc,'--groundTruth='+self.targ_dir,\
                        '--predictions='+self.pred_dir,\
                        '--evalPoseTracking'])
            except subprocess.CalledProcessError as e:
                sys.exit(e.output)

            pose_track = pose_track.decode().split('\n')[-4:]

            metric  = pose_track[0].replace('\\','').replace(' ','').split('&')[1:]
            headers = pose_track[1].replace('\\','').replace(' ','').split('&')[1:]
            scores  = pose_track[2].replace('\\','').replace(' ','').split('&')[1:]

            print(metric)
            print(headers)
            print(scores)

            if not self.debug:
                for m,h,s in zip(metric, headers, scores):
                    self.logger.log({m+' '+h:float(s)})

            print('Saved to: {}'.format(self.pred_dir))

            return ap_total 

        return 0

class Eval_PoseTrack17_det():
    """
    Write predictions for submission in JSON format. For PoseTrack17 submission.
    From bounding box detections, not ground truth
    """
    def __init__(self, *args, **kwargs):
        #self.result_dir = kwargs['result_dir']

        self.load_type  = kwargs['load_type']
        self.batch_size = kwargs['batch_size'] 
        self.ndata      = kwargs['ndata']
        self.count = 0

        self.correct = 0
        self.total   = 0

        self.json_anns = {}

        self.posetrack18_to_posetrack17 = [[0,13], [1,12], [2,14], [5,9], [6,8],
                                            [7,10], [8,7], [9,11], [10,6], [11,3],
                                            [12,2], [13,4], [14,1], [15,5], [16,0]]

	    #Any keypoints below this confidence threshold will be zero-ed out
        self.conf_threshold = kwargs['conf_threshold'] 

        #Run PoseTrack evaluation directly from here
        poseval_dir = kwargs.get('poseval_dir', '/z/home/natlouis/poseval/py')
        os.environ['PYTHONPATH'] = os.path.join(poseval_dir,'..','py-motmetrics:'+os.getenv('PYTHONPATH',''))

        #self.exec_loc = os.path.join(poseval_dir, 'evaluate.py')
        self.exec_loc = 'poseval.evaluate' #using faster poseval
        run_id = kwargs.get('run_id', kwargs['exp'])
        self.pred_dir = os.path.join(poseval_dir, 'prediction-'+kwargs['model']+'-'+run_id+'/')

        #TODO: Cheap and easy fix, shouldn't leave permanently like this
        tags = kwargs['tags']
        if tags:
            fold = tags[0]
            self.targ_dir = os.path.join(poseval_dir, 'temp_target_'+fold+'/')
        else:
            self.targ_dir = os.path.join(poseval_dir, 'target_2017/')
        
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.targ_dir, exist_ok=True)
        self.match_strategy = kwargs['match_strategy']
        self.l2_margin      = kwargs['l2_margin']
        self.spa_con_thresh = kwargs['spa_con_thresh']
        self.last_T_frames  = kwargs['last_T_frames']

        if self.match_strategy == 'gcn':
            self.gcn_margin  = kwargs['cont_acc_margin']
            gcn_model_ckpt   = kwargs['gcn_checkpoint']
            self.gcn_model   = GCN(in_channels=kwargs['in_channels'], edge_importance_weighting=True,\
                                    layout='hand', partition_strategy='spatial', gcn_vis_embed=kwargs['gcn_vis_embed']).to('cuda')
            self.gcn_model.load_state_dict(torch.load(gcn_model_ckpt)['state_dict'])
            self.gcn_model.eval()

        self.prev_vis_feat   = [] #saved video features for each object
        self.last_vis_feat   = {} #last tracked vis_features with each assigned track id

        self.prev_vid_id    = 'null'
        self.prev_image_id  = 'null'
        self.prev_frame     = None 
        self.prev_seq_name  = 'null'
        self.prev_keypoints = [] #saved keypoints from previous image
        self.prev_bbox      = [] #saved bbox detections for each object
        self.hungarian = Hungarian()

        self.viz = kwargs['viz']

        self.logger = kwargs['logger'] #wandb logging tool
        self.debug  = kwargs['debug']

    def get_max_preds(self, batch_heatmaps):
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

    def transform_pts(self, pt, t):
        '''
        Apply transform to predicted coordinates, w/ respect to original image
        '''
        new_pt = np.hstack((pt, np.ones((pt.shape[0],1))))
        new_pt = np.dot(t, new_pt.transpose()).transpose()

        return new_pt
    
    def keypoints_to_bbox(self, kpts, sc, f_size):
        '''
        Capture largest bounding box around predicted keypoints. Using for Spatial Consistency i.e. IoU 
            kpts (Tensor, shape [N,J,3] or [J,3])
            sc   (Tensor, shape []) : scale width and height
            f_size (Tuple, shape (2)): (width, height)
        '''
        assert sc > 0
        max_width  = f_size[0]*1.0
        max_height = f_size[1]*1.0

        bboxes = []

        if kpts.dim() < 3:
            kpts = kpts.unsqueeze(0)
        
        for kpt in kpts:
            mask = kpt[...,-1] > self.conf_threshold 

            xmin = torch.min(kpt[mask,0])
            ymin = torch.min(kpt[mask,1])
            xmax = torch.max(kpt[mask,0])
            ymax = torch.max(kpt[mask,1])

            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w * 0.5
            cy = ymin + h * 0.5

            if w == 0 or h == 0: #invalid bboxes will occupy entire screen, should induce low IoU
                w = max_width
                h = max_height
                
            #Enlarge by scale
            w *= sc
            h *= sc

            xmin = torch.clamp(cx - (0.5 * w), min=0)
            ymin = torch.clamp(cy - (0.5 * h), min=0)
            xmax = torch.clamp(cx + (0.5 * w), max=max_width)
            ymax = torch.clamp(cy + (0.5 * h), max=max_height)

            bboxes.append(torch.tensor([xmin, ymin, xmax, ymax]))

        return torch.stack(bboxes)

    def assign_ids(self, json_anns, seq_name, image_id, keypoints, det_bboxes, frame_dims, frame, viz=False, vis_feats=None, label=None):
        if len(keypoints) == 0:
            return #No keypoints to track

        frame_w, frame_h = frame_dims 
        #keypoints: list of Nx3 arrays (x,y,score)
        anns = []
        tids = {}

        kpts_curr_frame = torch.tensor(np.array(keypoints))
        bbox_curr_frame = torch.stack(det_bboxes)
        prev_anns = json_anns[seq_name]['annotations']

        if vis_feats:
            vis_feats_curr_frame = torch.stack(vis_feats)

        #Compare to poses from last T frames 
        T = self.last_T_frames
     
        if len(prev_anns) > 0:#Previous frame annotations exist
            tids_prev_frame = []
            kpts_prev_frame = []
            bbox_prev_frame = []

            uniq_frames = []
            for ann in reversed(prev_anns):
                uniq_frames.append(ann['image_id'])
                uniq_frames = list(set(uniq_frames))

                if len(uniq_frames) > T:
                    break

                if ann['image_id'] < image_id: #previous frame(s)
                    tids_prev_frame.append(ann['track_id'])
                    kpts_prev_frame.append(torch.tensor(ann['keypoints']).view(-1,3))
                    bbox_prev_frame.append(torch.tensor(ann['bbox']))

            #assign ids to previous T frames
            kpts_prev_frame = torch.stack(kpts_prev_frame)
            bbox_t          = torch.stack(bbox_prev_frame)

            try:
                if self.match_strategy == 'l2': #L2-Distance between points
                    margin = self.l2_margin

                    kpts_curr_frame = torch.tensor(keypoints)
                    M = kpts_curr_frame.shape[0]
                    N = kpts_prev_frame.shape[0]
                    costs = 1e10 * np.ones((M,N))
                    for i, kp in enumerate(torch.tensor(keypoints)):
                        dists = torch.mean(torch.sqrt((kpts_prev_frame[...,:2] - kp[:,:2]).pow(2).sum(2)),dim=1)
                        d = dists.shape[0]
                        costs[i,:d] = dists

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        dist = costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if dist < margin and _tid not in tids.values():
                            tids[tcurr] = _tid
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

                elif self.match_strategy == 'sc': #Spatial Consistency
                    threshold = self.spa_con_thresh

                    M = bbox_curr_frame.shape[0]
                    N = bbox_t.shape[0]
                    costs = 1e10 * np.ones((M,N))
                    #Get IOU between all pairs
                    for i, box_p in enumerate(bbox_curr_frame):
                        intersect_area = IOU().intersect(box_p, bbox_t) 

                        box_p_area = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
                        box_t_area = (bbox_t[:,2] - bbox_t[:,0]) * (bbox_t[:,3] - bbox_t[:,1])
                        union = box_p_area + box_t_area - intersect_area 

                        union = torch.clamp(union, min=0.001)
                        overlaps = intersect_area/union

                        d = overlaps.shape[0]
                        costs[i,:d] = 1 - overlaps

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        overlap = 1 - costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if overlap > threshold and _tid not in tids.values():
                            tids[tcurr] = _tid 
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

                elif self.match_strategy == 'gcn': #Graph Convolutional Network
                    margin = self.gcn_margin 

                    M = kpts_curr_frame.shape[0]
                    N = kpts_prev_frame.shape[0]
                    costs = 1e10 * np.ones((M,N), dtype=np.int32)

                    #Normalize keypoints to [0,1] w/ respect to detection box size
                    widths = bbox_curr_frame[:,2]-bbox_curr_frame[:,0]
                    heights= bbox_curr_frame[:,3]-bbox_curr_frame[:,1]
                    norm_kpts_curr = kpts_curr_frame - torch.cat((bbox_curr_frame[:,:2], torch.zeros(M,1)), dim=1).unsqueeze(1)
                    norm_kpts_curr /= torch.stack((widths, heights, torch.ones(M)),dim=1).unsqueeze(1)

                    widths = bbox_t[:,2]-bbox_t[:,0]
                    heights= bbox_t[:,3]-bbox_t[:,1]
                    norm_kpts_prev = kpts_prev_frame - torch.cat((bbox_t[:,:2], torch.zeros(N,1)), dim=1).unsqueeze(1)
                    norm_kpts_prev /= torch.stack((widths, heights, torch.ones(N)),dim=1).unsqueeze(1)

                    #Zero-out unannotated points
                    norm_kpts_curr[norm_kpts_curr[...,2] == 0] = -1
                    norm_kpts_prev[norm_kpts_prev[...,2] == 0] = -1

                    if vis_feats:
                        #Get corresponding visual features
                        vis_feats_last_frame = torch.zeros((N,vis_feats_curr_frame.shape[-1]))
                        for i,_tid in enumerate(tids_prev_frame):
                            vis_feats_last_frame[i] = self.last_vis_feat[_tid]

                        #Extract joint graph embedding for all keypoints + vis features
                        with torch.no_grad():
                            curr_embeds  = self.gcn_model.extract_joint_embed(vis_feats_curr_frame.to('cuda'), norm_kpts_curr[:,None,:,:2].to('cuda'))
                            prev_embeds  = self.gcn_model.extract_joint_embed(vis_feats_last_frame.to('cuda'), norm_kpts_prev[:,None,:,:2].to('cuda'))
                    else:
                        #Extract graph embedding for all keypoints
                        with torch.no_grad():
                            curr_embeds  = self.gcn_model.extract_feature(norm_kpts_curr[:,None,:,:2].to('cuda'))
                            prev_embeds  = self.gcn_model.extract_feature(norm_kpts_prev[:,None,:,:2].to('cuda'))

                    #Compute distance between all pairs
                    for i, ce in enumerate(curr_embeds):
                        dists = torch.sqrt((prev_embeds - ce).pow(2).sum(-1))
                        d = dists.shape[0]
                        costs[i,:d] = dists.cpu()

                    #Assign matches
                    try:
                        self.hungarian.calculate(costs)
                    except HungarianError:
                        print('Error. Not all matches found')
                    finally:
                        results = self.hungarian.get_results()

                    for (tcurr,tprev) in results:
                        dist = costs[tcurr,tprev]
                        _tid = tids_prev_frame[tprev]
                        if dist < margin and _tid not in tids.values():
                            tids[tcurr] = _tid
                        else:
                            tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)
            except BaseException as e:
                print('Error: {}'.format(e))
                print('image_id: {}'.format(image_id))
                import pdb; pdb.set_trace()

            #If many new objects are introduced, they will not get assigned from matching algorithm
            #Because there won't be a previous id to assign to. The costs matrix can be wide, but not tall
            #Happens a lot with cut-scenes
            if M > len(results):
                for tcurr in range(M):
                    if tcurr in tids: 
                        continue 

                    print('New object introduced at frame: {}'.format(image_id))
                    tids[tcurr] = (max(max(tids.values(), default=-1),max(tids_prev_frame)) + 1)

        else: #First frame, initialize all track ids
            for i in range(len(keypoints)):
                tids[i] = i

        assert(len(list(set(tids.values()))) == len(keypoints))

        self.last_vis_feat = {}
        for i,(kp,det_box) in enumerate(zip(keypoints, det_bboxes)):
            anns.append({'image_id':image_id,
                   'track_id': tids[i],
                   'keypoints':kp.reshape(-1).tolist(),
                   'bbox':det_box.reshape(-1).tolist(),
                   'scores':kp[:,-1].tolist()})

            if vis_feats:
                self.last_vis_feat[tids[i]] = vis_feats_curr_frame[i]  

        if viz:
            #must use original image
            img = frame.numpy()

            for i, bbox in enumerate(bbox_curr_frame.int()):
                xmin, ymin, xmax, ymax = bbox 

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color=(255,255,255), thickness=2)

                for idx,(p1,p2) in enumerate(self.links):
                    x1, y1, c1 = kpts_curr_frame[i, p1]
                    x2, y2, c2 = kpts_curr_frame[i, p2]

                    c = self.link_color[idx].tolist() #R,G,B
                    if c1 != 0:
                        cv2.circle(img, (x1,y1), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                    if c2 != 0 :
                        cv2.circle(img, (x2,y2), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                    if c1 != 0 and c2 != 0:
                        cv2.line(img, (x1,y1), (x2,y2), (c[2],c[1],c[0]), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'track id:'+str(tids[i]), (xmin, ymin), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

            #cv2.imshow('Output', img)
            cv2.imshow('Output', img[...,::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        json_anns[seq_name]['annotations'].extend(anns)

        return anns

    def get_accuracy(self, predictions, data):
        """
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

        Return:
            Accuracy # of correct case/ # of total cases
        """

        bbox         = data['bbox']
        #bbox         = data['input_crop']

        frame_path   = data['frame_path']
        frame_id     = data['frame_ids']
        nframes      = data['nframes']
        frame_size   = data['frame_size']
        vid_ids      = data['vid_id']

        neighbor_link = data['neighbor_link']
        link_colors   = data['link_colors']

        self.links = neighbor_link[0]
        self.link_color = link_colors[0]

        vis_feats = None
        if isinstance(predictions, tuple):
            _, _, _, _, _, out6 = predictions
            predictions = out6.unsqueeze(1)
        elif isinstance(predictions, dict):
            vis_feats   = predictions['feat']
            predictions = predictions['outputs']

        predictions = predictions.cpu().numpy()
        B,T,D,H,W = predictions.shape

        #Reshape, temporal dimension now represents multiple objects per image
        predictions = np.reshape(predictions, (B*T,D,H,W))
        idx = list(range(predictions.shape[1]))
        
        pred, maxvals = self.get_max_preds(predictions)
        #maxvals[:,3:5] = 0 #Left-Right ears are un-annotated
        pred_mask = maxvals > 0
        scores  = np.clip(maxvals,0,1)

        #if self.load_type == 'test' and self.batch_size == 1:
        cat = data['categories']
        categories = {}
        categories['name']      = cat['name'][0]
        categories['keypoints'] = [k[0] for k in cat['keypoints']]

        for bt in range(B*T):
            b = int(bt/T)
            t = bt % T

            f_path   = frame_path[0][b]
            frame_w  = frame_size[0][b]
            frame_h  = frame_size[1][b]
            image_id = frame_id[b,0].item()
            vid_id   = vid_ids[b]

            if f_path != 'None':
                seq_name = vid_id

            if bbox[b,t,0] == -1: #skipped detection (low score) or no detection
                is_labeled = False
                ann = None
                
                if t >= len(frame_path): #no frames at time t
                    continue
            else:
                is_labeled = True

                keypoints = np.concatenate((pred[bt], pred_mask[bt]), axis=-1)

                if 'stride' in data:
                    #Scale coordinates from heatmap size
                    sw, sh = data['stride'][b]
                    keypoints[:,0] *= sw.item() 
                    keypoints[:,1] *= sh.item()

                    #Apply inverse transform w/ respect original image
                    inv_trans = data['inv_trans']
                    keypoints[:,:2] = self.transform_pts(keypoints[:,:2], inv_trans[b,t])
                    keypoints[:,-1] = scores[bt].squeeze() #Add scores to keypoints
                    thresh = 0
                    keypoints[keypoints[:,2] < thresh] = 0

                else:
                    input_crop = data.get('input_crop', data['bbox'])
                    padding    = data.get('padding', torch.zeros(B,T,4))

                    crop = input_crop[b,t]
                    pad  = padding[b,t] #pl, pt, pr, pb

                    #scale coordinates to crop size
                    crop_h = (crop[3]-crop[1]).item()
                    crop_w = (crop[2]-crop[0]).item()
                    keypoints[:,0] *= (crop_w/W)
                    keypoints[:,1] *= (crop_h/H)

                    #Undo crop
                    keypoints[:,0] += crop[0].item()
                    keypoints[:,1] += crop[1].item()

                    #Subtract padding if was added
                    keypoints[:,0] -= pad[0].item()
                    keypoints[:,1] -= pad[1].item()

                    keypoints[:,-1] = scores[bt].squeeze() #Add scores to keypoints

                #if np.sum(keypoints[:,-1]) == 0: #Skip poses w/ no valid keypoints
                #    continue 
                
                temp_score = np.copy(scores[bt].reshape(-1))

                #Zero-out values below confidence threshold
                mask = temp_score < self.conf_threshold
                temp_score[mask] = 0
                keypoints[mask] = 0

                #Filter keypoints outside of tight bounding box
                xmin, ymin, xmax, ymax = bbox[b,t]
                for j in range(D):
                    x1, y1, _ = keypoints[j]
                    if x1 < xmin or x1 > xmax or y1 < ymin or y1 > ymax:
                        keypoints[j] = 0

                if 'data' in data:
                    frame = data['data'][b]
                else:
                    frame = None

                #Assigning Track Id
                if self.prev_vid_id != vid_id: #New video
                    if self.prev_seq_name != 'null':
                        self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.prev_frame, self.viz, self.prev_vis_feat)

                    self.prev_vid_id    = vid_id
                    self.prev_image_id  = image_id 
                    self.prev_seq_name  = seq_name
                    self.prev_frame     = frame 
                    self.prev_keypoints = []
                    self.prev_bbox      = []
                    self.prev_vis_feat  = []

                    json_data = {'annotations':[], 'categories':[categories]}    
                    self.json_anns[seq_name] = json_data 

                elif self.prev_image_id != image_id: #Same video, new frame
                    self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), self.prev_frame, self.viz, self.prev_vis_feat)
                    
                    self.prev_image_id = image_id
                    self.prev_frame     = frame 
                    self.prev_keypoints = []
                    self.prev_bbox      = []
                    self.prev_vis_feat  = []

                if np.sum(keypoints[...,-1]>self.conf_threshold) >= 1: #Only add if atleast 1 keypoint is above threshold
                    self.prev_keypoints.append(keypoints)
                    self.prev_bbox.append(bbox[b,t].float())
                    if vis_feats is not None:
                        self.prev_vis_feat.append(vis_feats[b,t])

        self.count += B

        if self.count >= self.ndata:
            #Last frame
            self.assign_ids(self.json_anns, self.prev_seq_name, self.prev_image_id, self.prev_keypoints, self.prev_bbox, (frame_w, frame_h), frame, self.viz, self.prev_vis_feat)

            for seq_name, json_data in self.json_anns.items():
                #Add images from gt annotation b/c not all frames labeled and annotated 
                gt_json_file = os.path.join(self.targ_dir, seq_name+'.json')
                
                with open(gt_json_file, 'r') as f:
                    gt_json_data = json.load(f)

                images = []
                for item in gt_json_data['annolist']:
                    imgnum = item['imgnum'][0]
                    frame_id = int('1'+seq_name.split('_')[0]+str(imgnum).zfill(4))

                    images.append({'file_name':item['image'][0]['name'], 'id':frame_id})
                        
                json_data['images'] = images 
                json_file = os.path.join(self.pred_dir, seq_name+'.json')

                with open(json_file, 'w') as f:
                    json.dump(json_data, f)

            print('Running PoseTrack Pose Estimation Eval')
            pose_est = subprocess.check_output(['python', '-m', self.exec_loc,'--groundTruth='+self.targ_dir,\
                    '--predictions='+self.pred_dir,\
                    '--evalPoseEstimation'])

            pose_est = pose_est.decode().split('\n')[-4:]

            metric  = pose_est[0]
            headers = pose_est[1].replace('\\','').replace(' ','').split('&')[1:]
            scores  = pose_est[2].replace('\\','').replace(' ','').split('&')[1:]

            ap_total = float(scores[-1])*.01

            print(metric)
            print(headers)
            print(scores)

            if not self.debug:
                for h,s in zip(headers, scores):
                    self.logger.log({'AP '+h:float(s)})

            print('Running PoseTrack Pose Tracking Eval')
            pose_track = subprocess.check_output(['python', '-m', self.exec_loc,'--groundTruth='+self.targ_dir,\
                    '--predictions='+self.pred_dir,\
                    '--evalPoseTracking'])

            pose_track = pose_track.decode().split('\n')[-4:]

            metric  = pose_track[0].replace('\\','').replace(' ','').split('&')[1:]
            headers = pose_track[1].replace('\\','').replace(' ','').split('&')[1:]
            scores  = pose_track[2].replace('\\','').replace(' ','').split('&')[1:]

            print(metric)
            print(headers)
            print(scores)

            if not self.debug:
                for m,h,s in zip(metric, headers, scores):
                    self.logger.log({m+' '+h:float(s)})

            print('Saved to: {}'.format(self.pred_dir))

            return ap_total 

        return 0
