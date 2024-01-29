#Execute a single video sequence from a saved checkpoint
import os
import glob
import sys
DIRECTORY = os.getcwd()
sys.path.append(DIRECTORY)

import yaml
import argparse
import json
import cv2

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from scipy.signal import medfilt

from models.models_import import create_model_object
from datasets import data_loader
from checkpoint import load_checkpoint

from tqdm import tqdm

#torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

parser.add_argument('--cfg_file', type=str, help='Link to YAML file for configuration of saved run')

#parameters to override
parser.add_argument('--dataset',        type=str,   default='Hand_Dets')
parser.add_argument('--json_path',      type=str, help='Input JSON file of bounding box detections in ViP format')
parser.add_argument('--exp',            type=str, help='Experiment/run name')
parser.add_argument('--checkpoint',     type=str, help='Pretrained checkpoint for model')
parser.add_argument('--clip_length',    type=int,   default=-1, help='Export entire video')
parser.add_argument('--num_clips',      type=int,   default=1, help='Export entire video')
parser.add_argument('--load_type',      type=str,   default='runtime')
parser.add_argument('--shuffle_train',  type=bool,  default=False)
parser.add_argument('--conf_threshold', type=float, default=0.0, help='Keypoint threshold')

#Runtime specific parameters
parser.add_argument('--num_gpus', type=int, default=-1)
parser.add_argument('--viz', action='store_true', help='Show visualization')
parser.add_argument('--write_vid', action='store_true', help='Save overlayed video to save_dir')
parser.add_argument('--save_dir', type=str, default='./results_runtime/', help='Save data here')

cmd_args,extra_args = parser.parse_known_args()
cmd_args = vars(cmd_args)

#Grab extra args passed through command line
for item in extra_args:
    try:
        name, value = item.split('--')[-1].split('=')
    except ValueError:
        sys.exit('{} must be in format --ARG=VALUE'.format(item))

    #Need better way to automatically typecast variables
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                value = json.loads(value.lower())
            else:
                #Maybe it is a list
                if ',' in value or '[' in value:
                    value = value.replace('[','').replace(']','').split(',')
                else:
                    value = str(value) #Keep as string

    #Add to global args list
    cmd_args[name] = value

#Load configuration file
with open(cmd_args['cfg_file']) as f:
    yaml_args = yaml.safe_load(f)

#Update args with cmd line args if exists
for (k,v) in cmd_args.items():
    if v is not None:
        yaml_args[k] = cmd_args[k]

print('Running with following configs')
for (k,v) in yaml_args.items():
    print('{}: {}'.format(k,v))
print('--'*30)

run_id     = yaml_args['cfg_file'].split('/')[-2].split('_')[-1]
exp        = yaml_args['exp']
model_name = yaml_args['model']

save_dir   = yaml_args['save_dir']
write_vid  = yaml_args['write_vid']
visualize  = yaml_args['viz']

os.makedirs(save_dir, exist_ok=True)

class Runtime:

    def __init__(self, **kwargs):
        #Model config
        model = create_model_object(**kwargs)
        print('Using model: {}'.format(type(model).__name__))
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        #ckpt_path = glob.glob(os.path.join(*kwargs['cfg_file'].split('/')[:-1],'checkpoints','*_best_model.pkl'))[0]
        ckpt_path = kwargs['pretrained']
        print('Loading checkpoint from: {}'.format(ckpt_path))
        ckpt = load_checkpoint(ckpt_path)

        ckpt_keys = list(ckpt.keys())
        if ckpt_keys[0].startswith('module.'): #if checkpoint weights are from DataParallel object
            for key in ckpt_keys:
                ckpt[key[7:]] = ckpt.pop(key)

        model.load_state_dict(ckpt)
        model.to(self.device) 

        self.loader = data_loader(model_obj=model, **kwargs)[kwargs['load_type']]
        self.model = model

    ##DEPRECATED
    def execute(self, input_seq, aux_data=None):
        with torch.no_grad():
            input_seq = input_seq.to(self.device)
            B,T,D = input_seq.shape
            out = []

            if aux_data is not None:
                aux_data = aux_data.to(self.device)
                for i in range(self.pad, T - self.pad):
                    out.append(self.model(input_seq[:,i-self.pad:i+self.pad], aux_data[:,i-self.pad:i+self.pad]))
            else:
                for i in range(self.pad, T - self.pad):
                    out.append(self.model(input_seq[:,i-self.pad:i+self.pad]))
            
            return out

def get_max_preds(batch_heatmaps):
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

#remove jittery keypoints by applying a median filter along each axis
def median_filter_kpts(kpts, window_size = 3): 
    filtered = copy.deepcopy(kpts)

    #apply median filter to get rid of poor keypoints estimations
    for j in range(kpts.shape[1]):
        xs = kpts[:,j,0]
        ys = kpts[:,j,1]

        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)

        if kpts.shape[-1] == 3:
            zs = kpts[:,j,2]
            zs = medfilt(zs, window_size)
            filtered[:,j] = np.stack([xs, ys, zs], axis = -1) 
        else:
            filtered[:,j] = np.stack([xs, ys], axis = -1) 

    return filtered

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0) 
          ):  

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

if __name__ == "__main__":

    runtime = Runtime(**yaml_args)

    conf_threshold = yaml_args['conf_threshold']
    outputs = {}
    with torch.no_grad():
        for data in tqdm(runtime.loader):
            x_input = data['data']
            annots  = data['annots']

            bbox          = annots['bbox']
            frame_path    = annots['frame_path']
            frame_size    = annots['frame_size']
            input_crop    = annots.get('input_crop', annots['bbox']) 
            vid_id        = annots['vid_id']
            tids          = annots['track_ids']
            neighbor_link = annots['neighbor_link']
            link_colors   = annots['link_colors']
            labels        = annots.get('labels', None)
            fps           = annots['fps']

            if isinstance(x_input, torch.Tensor):
                predictions = runtime.model(x_input.to(runtime.device))
            else:
                for i, item in enumerate(x_input):
                    if isinstance(item, torch.Tensor):
                        x_input[i] = item.to(runtime.device)
                predictions = runtime.model(*x_input)

            B,T,D,H,W = predictions.shape
            padding   = annots.get('padding', torch.zeros(B,4)).squeeze()

            predictions = predictions.cpu().numpy()
            predictions = np.reshape(predictions, (B*T,D,H,W)) 
            input_crop  = np.reshape(input_crop, (B,-1))

            pred, maxvals = get_max_preds(predictions)
            pred_mask = maxvals > conf_threshold
            scores = np.clip(maxvals,0,1)
            links = neighbor_link[0]
            link_color = link_colors[0]

            pred = np.reshape(pred, (B,D,2))
            pred_mask = np.reshape(pred_mask, (B,D,1))

            for b in range(B):
                t=0
                frame_w = frame_size[0][0]
                frame_h = frame_size[1][0]
                box    = bbox[b,t].numpy()
                crop   = input_crop[b]
                pad    = padding[b] #pl, pt, pr, pb

                seq_name = vid_id[b]
                f_path   = seq_name+'/'+frame_path[t][b].split('/')[-1]
                root_dir = '/'.join((frame_path[t][b].split('/')[:-2]))
                if seq_name not in outputs.keys():
                    outputs[seq_name] = {'frame_width':frame_w.item(), 'frame_height':frame_h.item(), 'fps':fps[b].item(), 'root_dir':root_dir, 'frames':{}}
                if f_path not in outputs[seq_name]['frames'].keys():
                    outputs[seq_name]['frames'][f_path] = {'objs':[]}

                if crop[0] == -1:
                    continue

                kpts = np.concatenate((pred[b], pred_mask[b]), axis=-1)

                #scale coordinates to crop size
                crop_h = (crop[3]-crop[1]).item()
                crop_w = (crop[2]-crop[0]).item()
                kpts[:,0] *= (crop_w/W)
                kpts[:,1] *= (crop_h/H)

                #Undo crop
                kpts[:,0] += crop[0].item()
                kpts[:,1] += crop[1].item()

                #Subtract padding if was added
                kpts[:,0] -= pad[0].item()
                kpts[:,1] -= pad[1].item()
                
                if np.sum(kpts[...,-1]>conf_threshold) >= 1:
                    outputs[seq_name]['frames'][f_path]['objs'].append({
                                                    'box':box.tolist(),
                                                    'kpts':kpts.tolist(),
                                                    'tid':tids[b,t].item(),
                                                   })
    
    #Filter bounding boxes and keypoints
    for seq_name in outputs.keys():
        tracked_objs = {}

        #Collect bounding boxes and keypoints
        for frame_path in sorted(outputs[seq_name]['frames'].keys()):
            frame_dat = outputs[seq_name]['frames'][frame_path]
            objs = frame_dat['objs']
            for obj in objs:
                box  = np.array(obj['box'])
                kpts = np.array(obj['kpts'])
                tid  = obj['tid']

                if tid not in tracked_objs:
                    tracked_objs[tid] = {'box':[],
                                         'kpts':[],
                                         'frame_paths':[]
                                        }

                tracked_objs[tid]['box'].append(box)
                tracked_objs[tid]['kpts'].append(kpts)
                tracked_objs[tid]['frame_paths'].append(frame_path)

        #Apply filter
        wsize = 5
        for tid in tracked_objs.keys():
            box  = np.array(tracked_objs[tid]['box'])
            kpts = np.array(tracked_objs[tid]['kpts'])
            frame_paths = tracked_objs[tid]['frame_paths']

            if len(box) > wsize:
                kpts_filtered = median_filter_kpts(kpts, wsize) 

                #Place back into outputs
                for idx,frame_path in enumerate(frame_paths): 
                    frame_dat = outputs[seq_name]['frames'][frame_path]
                    objs = frame_dat['objs']
                    for obj in objs:
                        if obj['tid'] == tid:
                            obj['kpts'] = kpts_filtered[idx].tolist()


    for seq_name in outputs.keys():
        #Save results to a file
        json_file = os.path.join(save_dir, seq_name+'.json')
        with open(json_file, 'w') as f:
            json.dump(outputs[seq_name], f)
        print('Wrote file to: {}'.format(json_file))

    if write_vid or visualize: 
        tid_clr   = {}
        track_colors = [
                        (55, 168, 99),
                        (130, 20, 200),
                        (123, 45, 67),
                        (210, 87, 32),
                        (78, 200, 90),
                        (44, 189, 240),
                        (255, 150, 10),
                        (20, 120, 255),
                        (240, 50, 30),
                        (160, 70, 180)
                        ]

        for seq_name in outputs.keys():
            frame_width  = outputs[seq_name]['frame_width'] 
            frame_height = outputs[seq_name]['frame_height']
            fps          = outputs[seq_name]['fps']
            vid_path = None
            if write_vid:
                out_dir = os.path.join(save_dir, model_name, 'videos')
                os.makedirs(out_dir, exist_ok=True)

                vid_path = os.path.join(out_dir,seq_name+'.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(vid_path, fourcc, fps, (frame_width, frame_height))

            root_dir = outputs[seq_name]['root_dir']
            #Draw detections on frames
            for frame_path in tqdm(sorted(outputs[seq_name]['frames'].keys())):
                frame_dat = outputs[seq_name]['frames'][frame_path]

                #Load frame image data and swap color channels to RGB
                img = cv2.imread(os.path.join(root_dir,frame_path))

                box_line_size = 1 + np.floor(frame_width/400).astype('int')
                line_size = 3 * max(1, np.floor(frame_width/400).astype('int'))
                rad_size  = 2 * max(1, np.floor(frame_width/400).astype('int'))
                font = cv2.FONT_HERSHEY_SIMPLEX
                objs = frame_dat['objs']

                track_idx = 0
                for obj in objs:
                    box  = np.array(obj['box'])
                    kpts = np.array(obj['kpts']) 
                    tid  = obj['tid']

                    if tid not in tid_clr.keys():
                        tid_clr[tid] = track_colors[track_idx%len(track_colors)]
                        track_idx += 1
                    box_color = tid_clr[tid]

                    xmin,ymin,xmax,ymax = map(int, box)
                    #Draw bounding box
                    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), color=box_color, thickness=box_line_size)

                    #Draw joints
                    for idx,(p1,p2) in enumerate(links):
                        x1,y1,c1 = kpts[p1]
                        x2,y2,c2 = kpts[p2]

                        c = link_color[idx].tolist() #R,G,B
                        if c1 != 0:
                            cv2.circle(img, (int(x1),int(y1)), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                        if c2 != 0 :
                            cv2.circle(img, (int(x2),int(y2)), radius=5, color=(c[2],c[1],c[0]), thickness=-1)
                        if c1 != 0 and c2 != 0:
                            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (c[2],c[1],c[0]), 3)

                    #Draw tracking id
                    cv2.putText(img, 'id: '+str(tid), (int(xmin), int(ymin)), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

                draw_text(img, frame_path.split('/')[-1], pos=(0,0),
                          text_color=(255,255,255), text_color_bg=(0,0,0))

                if write_vid:
                    video_writer.write(img)
                else:
                    cv2.imshow('Output', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            #Release the video writer
            if write_vid:
                video_writer.release()
                print('Wrote video out to: {}'.format(vid_path))
