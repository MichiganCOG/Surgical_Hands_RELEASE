#Compile all JSON files and convert to the Detection format needed
#for the ViP dataset
#mixed between real and synth datasets

import json
import os 
import glob

import numpy as np
from PIL import Image

#Order here matters
splits = ['train', 'synth1', 'synth2', 'synth3', 'synth4', 'test']
#The synths will all be compiled into train.json

target_dir = 'data/hand_labels_mixed'
os.makedirs(target_dir, exist_ok=True)

json_ann = []
for split in splits:
    if 'synth' in split:
        source_dir = 'data/hand_labels_synth/'+split
        target_file = os.path.join(target_dir,'train.json')
    else:
        source_dir = 'data/hand_labels/manual_'+split 
        target_file = os.path.join(target_dir,split+'.json')

    print('Compiling files from: {}'.format(source_dir))
    json_files = sorted(glob.glob(os.path.join(source_dir,'*.json')))

    #Gather data into bounding boxes
    for annot in json_files:
        frames = []
        frame = {} #An extra loop would've been required for a video dataset

        img_path = '.'.join((annot.split('.json')[0],'jpg'))
        base_path = '/'.join(img_path.split('/')[:-1])

        im = Image.open(img_path)
        width, height = im.size

        with open(annot, 'r') as f:
            data = json.load(f)
        
        objs       = []
        track_id   =  0 #only one hand per image 
        hand_pts   = data['hand_pts']
        head_size  = 0 if 'head_size' not in data else data['head_size']
        class_name = 'left' if data['is_left'] else 'right' 
        is_left    = data['is_left']
        occluded   = (1-np.array(hand_pts, dtype=np.int32)[:,2]).tolist()

        points    = np.array(hand_pts)
        valid_idx = np.argwhere(points[:,2] == 1)[:,0]

        xmin, ymin, _ = np.min(points[valid_idx], 0)
        xmax, ymax, _ = np.max(points[valid_idx], 0)

        objs.append({'trackid':track_id, 'c':class_name, 'occ': occluded, 'bbox':[float(xmin), float(ymin), float(xmax), float(ymax)], 'hand_pts':hand_pts, 'head_size':head_size, 'is_left':is_left})

        img_path = img_path.replace('/z','') #Keeping images local
        base_path = base_path.replace('/z','') #Keeping images local

        frame['objs'] = objs
        frame['img_path'] = img_path

        frames.append(frame)
        json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

    if split == 'synth4' or split == 'test':
        #Write out to CSV file
        print('Saving to: {}'.format(target_file))
        with open(target_file, 'w') as f:
            json.dump(json_ann, f)
        json_ann = []

