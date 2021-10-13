###
# Image crops come from bounding box detections
# Model: FlowTrack_R, Dataset: PoseTrack18_obj_det (or equi), Metric: Eval_PoseTrack18_det (or equi)
# Perform eval on a videos frames, sequentially. Important that output from previous time step
# can be used for current time step.
# Cycle through each prior individually, rather than max of all priors
# Expects that all image crops are sampled individually from images.
# low batch size, memory constraints and maybe 1 worker
# num_clips 0 and clip_length 1 
#
# Transform previous heatmaps based on current crop
# but don't regenerate the produced heatmaps
###

import os
import sys
import datetime
import yaml
import torch
import torch.nn.functional as F

import numpy                    as np
import cv2

import torch.nn                 as nn
import torch.optim              as optim
import torch.utils.data         as Data

from tensorboardX                       import SummaryWriter

from parse_args                         import Parse
from models.models_import               import create_model_object
from datasets                           import data_loader 
from metrics                            import Metrics
from checkpoint                         import load_checkpoint

import pprint

import wandb 

import kornia

def eval(**args):
    """
    Evaluate selected model 
    Args:
        seed       (Int):        Integer indicating set seed for random state
        save_dir   (String):     Top level directory to generate results folder
        model      (String):     Name of selected model 
        dataset    (String):     Name of selected dataset  
        exp        (String):     Name of experiment 
        load_type  (String):     Keyword indicator to evaluate the testing or validation set
        pretrained (Int/String): Int/String indicating loading of random, pretrained or saved weights
        
    Return:
        None
    """

    print("Experimental Setup: ")
    pprint.PrettyPrinter(indent=4).pprint(args)

    d          = datetime.datetime.today()
    date       = d.strftime('%Y%m%d-%H%M%S')
    result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],args['exp'],date)))
    log_dir    = os.path.join(result_dir, 'logs')
    save_dir   = os.path.join(result_dir, 'checkpoints')

    run_id = args['exp']
    if not args['debug']:
        wandb.init(project=args['dataset'], name=args['exp'], config=args, tags=args['tags'])

        #Replace result dir with wandb unique id, much easier to find checkpoints
        run_id = wandb.run.id
        if run_id:
            result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'], run_id)))
            log_dir    = os.path.join(result_dir, 'logs')
            save_dir   = os.path.join(result_dir, 'checkpoints')

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(log_dir,    exist_ok=True) 
        os.makedirs(save_dir,   exist_ok=True) 

        # Save copy of config file
        with open(os.path.join(result_dir, 'config.yaml'),'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

        # Tensorboard Element
        writer = SummaryWriter(log_dir)

    # Check if GPU is available (CUDA)
    num_gpus = args['num_gpus']
    device = torch.device("cuda:0" if num_gpus > 0 and torch.cuda.is_available() else "cpu")
    print('Using {}'.format(device.type))

    # Load Network
    model = create_model_object(**args).to(device)
    model_obj = model 

    if device.type == 'cuda' and num_gpus > 1:
        device_ids = list(range(num_gpus)) #number of GPUs specified
        model = nn.DataParallel(model, device_ids=device_ids)
        model_obj = model.module #Model from DataParallel object has to be accessed through module

        print('GPUs Device IDs: {}'.format(device_ids))

    # Load Data
    loader = data_loader(**args, model_obj=model_obj)

    if args['load_type'] == 'train_val':
        eval_loader = loader['valid']

    elif args['load_type'] == 'train':
        eval_loader = loader['train']

    elif args['load_type'] == 'test':
        eval_loader  = loader['test'] 

    else:
        sys.exit('load_type must be valid or test for eval, exiting')

    # Save dataset and metric Python file 
    if not args['debug']:
        dataset_file = os.path.join('datasets',eval_loader.dataset.__class__.__name__+'.py')
        wandb.save(dataset_file)

        dataset_file = os.path.join('metrics.py')
        wandb.save(dataset_file)

    if isinstance(args['pretrained'], str):
        ckpt = load_checkpoint(args['pretrained'])

        ckpt_keys = list(ckpt.keys())
        if ckpt_keys[0].startswith('module.'): #if checkpoint weights are from DataParallel object
            for key in ckpt_keys:
                ckpt[key[7:]] = ckpt.pop(key)

        model_obj.load_state_dict(ckpt, strict=False)

    # Training Setup
    params     = [p for p in model.parameters() if p.requires_grad]

    acc_metric = Metrics(**args, result_dir=result_dir, ndata=len(eval_loader.dataset), logger=wandb, run_id=run_id)
    acc = 0.0

    # Setup Model To Evaluate 
    model.eval()
    seq_data = {'vid_id':-1}

    with torch.no_grad():
        for step, data in enumerate(eval_loader):
            x_input     = data['data']
            annotations = data['annots']

            B,_,T,H,W = x_input.shape #Expect: B,3,T,384,288 (T here is not time, it's all of the objects detected on that frame)

            outputs = []
            feats   = []

            uniq_frame_ids = list(set([a.item() for a in annotations['frame_ids']]))
            for b in range(B):
                vid_id   = annotations['vid_id'][b]
                frame_id = annotations['frame_ids'][b].item()

                curr_crop  = annotations['input_crop'][b]
                frame_size = annotations['raw_frame_size'][b]

                heatmap_size = args['heatmap_size']
                num_joints   = args['num_joints']

                #print('frame id: {}'.format(frame_id))
                if seq_data['vid_id'] != vid_id:
                    print('New vid id: {}'.format(vid_id))

                    seq_data = {} #No need to save between sequences
                    seq_data['vid_id'] = vid_id
                    seq_data['prev_heatmap'] = {}
                    last_frame_id = None 
                    flag_new_frame = False 

                    torch.cuda.empty_cache()

                output = []
                image_crops = x_input[b].permute(1,0,2,3).to(device) #from (B,3,T,H,W) to (T,3,H,W), operate on each crops as mini-batch

                if args['model'] == 'FlowTrack':
                    out = model.forward(image_crops.unsqueeze(2)).squeeze(1)

                    if isinstance(out, tuple):
                        heatmap   = out[0] 
                        vis_feat  = out[1]
                    else:
                        heatmap = out 
                else:

                    prev_heatmap = []
                    if not last_frame_id is None and last_frame_id != frame_id:
                        flag_new_frame = True 

                    if last_frame_id is None or not flag_new_frame: #First frame of video
                        prev_hms = [] 
                    elif last_frame_id != frame_id: #New frame of video
                        prev_hms = seq_data['prev_heatmap'][last_frame_id]

                        #Delete other prev_heatmaps
                        for key in list(seq_data['prev_heatmap'].keys()):
                            if key not in [frame_id, last_frame_id]:
                                del seq_data['prev_heatmap'][key] 

                    #############################
                    if len(prev_hms) > 0 and 'stride' in annotations:
                        batch, joints, height, width = prev_hms.shape
                        input_height, input_width = args['final_shape']
                        prev_hms = kornia.warp_affine(prev_hms.to(device), annotations['trans'][b].repeat(batch,1,1).float().to(device),\
                                                   dsize=(input_height, input_width), align_corners=True)
                        prev_heatmap = F.interpolate(prev_hms, size=(heatmap_size[1], heatmap_size[0]))

                        #No point in adding prior if it's all zeros
                        keep_prior = torch.max(prev_heatmap.view(batch,-1), dim=-1)[0] != 0
                        prev_heatmap = prev_heatmap[keep_prior]
                        #torch.cuda.empty_cache() #clear memory of deleted tensors
                    else:
                        for hm_ in prev_hms:
                            scr = torch.mean(torch.max(hm_.contiguous().view(num_joints, -1), dim=-1)[0],dim=0)

                            if scr < args['min_gauss_peak_eval']:
                                prev_heatmap.append(torch.zeros(num_joints, heatmap_size[1], heatmap_size[0]).to(device))
                            else:
                                pl,pt,pr,pb = annotations['padding'][b,0].tolist() #current image padding
                                x1,y1,x2,y2 = curr_crop[0].int().tolist() #current image hand crop
                                #add current image padding
                                pad_tensor = nn.ConstantPad2d((pl,pr,pt,pb), 0.0)  #pad_left, pad_right, pad_top, pad_bot
                                hm_ = pad_tensor(hm_) #current hand crop w/ padding (only right and bottom padding need to be added)
                                #temp1 = hm_.clone()
                                hm_ = hm_[:,int(y1):int(y2),int(x1):int(x2)] #current crop position
                                #temp2 = hm_.clone()
                                hm_ = F.interpolate(hm_[:,None], size=heatmap_size)[:,0] #resized to heatmap size

                                #No point in adding prior if it's all zeros
                                if torch.max(hm_) > 0:
                                    prev_heatmap.append(hm_)
                        if len(prev_heatmap) > 0:
                            prev_heatmap = torch.stack(prev_heatmap).to(device)
                    #############################

                    num_priors = 1
                    if len(prev_heatmap) > 0:
                        '''
                        import matplotlib.pyplot as plt
                        for p_idx in range(len(prev_heatmap)):
                            plt.subplot(3,3,p_idx+1)
                            plt.imshow(torch.max(prev_heatmap[p_idx], dim=0)[0].cpu().numpy())
                            plt.title('Prior {}'.format(p_idx))
                        plt.show()
                        '''

                        num_priors   = prev_heatmap.shape[0]
                        image_crops = image_crops.repeat(num_priors,1,1,1)
                    else:
                        prev_heatmap = None 
                
                    out = model.forward_one(image_crops, prev_heatmap, {'frame_id': frame_id, 'batch_num':b}) 

                    if isinstance(out, tuple):
                        heatmap   = out[0] 
                        vis_feat  = out[1]
                    else:
                        heatmap = out 

                    #If multiple priors, pick prediction with highest average confidence
                    if num_priors > 1:
                        #max_conf_idx = torch.argmax(torch.mean(heatmap, dim=[1,2,3]))

                        _heatmap = heatmap.view(num_priors, heatmap.shape[1], -1)
                        max_conf_idx = torch.argmax(torch.mean(torch.max(_heatmap, dim=-1)[0],dim=1))

                        heatmap = heatmap[None,max_conf_idx]

                        if isinstance(out, tuple):
                            vis_feat = vis_feat[None, max_conf_idx]

                    ####Un-project output heatmap onto full frame before storing
                    #makes it easier to use for the next frame's bbox crop
                    x1,y1,x2,y2 = curr_crop[0].int().tolist()
                    img_width, img_height = frame_size
                    crop_h = y2 - y1
                    crop_w = x2 - x1
                    if 'stride' in annotations: #Human pose
                        input_height, input_width = args['final_shape']

                        hm_ = F.interpolate(heatmap, size=(input_height, input_width)) #resize from heatmap to input dimensions
                        hm_ = kornia.warp_affine(hm_.cpu(), annotations['inv_trans'][b].float(), dsize=(int(img_width), int(img_height)), align_corners=True) #inverse transform to full image
                    else: #Hand pose
                        pl,pt,pr,pb = annotations['padding'][b,0].tolist()
                        hm_ = F.interpolate(heatmap, size=(crop_h,crop_w))

                        pad_tensor = nn.ConstantPad2d((x1, max(0, ((img_width+pl+pr)-x2)), y1, max(0, ((img_height+pt+pb)-y2))), 0.0)  #pad_left, pad_right, pad_top, pad_bot
                        hm_ = pad_tensor(hm_) #prior hand crop reprojected onto full frame

                        _pb = (img_height+pt) if not pb else -pb #check if non-zero, and adjust for array slicing
                        _pr = (img_width+pl) if not pr else -pr 

                        hm_ = hm_[:,:,pt:_pb,pl:_pr] #prior hand crop w/o padding
                    ########

                    if frame_id in seq_data['prev_heatmap']:
                        seq_data['prev_heatmap'][frame_id] = torch.cat((seq_data['prev_heatmap'][frame_id], hm_))
                    else:
                        seq_data['prev_heatmap'][frame_id] = hm_

                last_frame_id = frame_id 

                outputs.append(heatmap) #append (T,D,H_,W_)
                if isinstance(out, tuple):
                    feats.append(vis_feat)

            outputs = torch.stack(outputs)
            if len(feats) > 0:
                feats = torch.stack(feats)

                outputs = {'outputs':outputs, 'feat':feats}

            if args['save_feat']:
                feats       = outputs['feat'].cpu().data
                gt_key_pts  = annotations['key_pts']
                obj_ids     = annotations['obj_ids']
                track_ids   = annotations['track_ids']
                vid_id      = annotations['vid_id']
                load_type   = annotations['load_type'][0]

                feat_dir = os.path.join(args['save_feat_dir'], args['model']+'-'+args['exp'], load_type)
                os.makedirs(feat_dir, exist_ok=True)

                for vid in set(vid_id):
                    idx = [i for i, item in enumerate(vid_id) if item == vid]

                    feat    = feats[idx]
                    key_pts = gt_key_pts[idx]
                    track   = track_ids[idx]
                    oid     = obj_ids[idx]

                    filename = os.path.join(feat_dir,vid+'.pkl') 
                    if os.path.exists(filename):
                        vid_data = torch.load(filename)
                        vid_data['feat']       = torch.cat((vid_data['feat'], feat))
                        vid_data['gt_key_pts'] = torch.cat((vid_data['gt_key_pts'], key_pts))
                        vid_data['track_id']   = torch.cat((vid_data['track_id'], track))
                        vid_data['object_ids'] = torch.cat((vid_data['object_ids'], oid))
                    else:
                        vid_data = {'feat':feat, 'gt_key_pts':key_pts, 'track_id':track, 'object_ids':oid}

                    torch.save(vid_data, filename)

                outputs = outputs['outputs']

            acc = acc_metric.get_accuracy(outputs, annotations)

            if step % 100 == 0:
                print('Step: {}/{} | {} acc: {:.4f}'.format(step, len(eval_loader), args['load_type'], acc))

    print('Accuracy of the network on the {} set: {:.3f} %\n'.format(args['load_type'], 100.*acc))

    if not args['debug']:
        writer.add_scalar(args['dataset']+'/'+args['model']+'/'+args['load_type']+'_accuracy', 100.*acc)
        # Close Tensorboard Element
        writer.close()

def crop_coords(x, y, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        if np.any(x > crop_xmax) or np.any(x < crop_xmin) or np.any(y > crop_ymax) or np.any(y < crop_ymin):
            return -1*np.ones(x.shape), -1*np.ones(y.shape)

        x_new = np.clip(x, crop_xmin, crop_xmax)
        y_new = np.clip(y, crop_ymin, crop_ymax)

        return x_new-crop_xmin, y_new-crop_ymin

if __name__ == '__main__':

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    eval(**args)
