#Compile all JSON files and convert to the Detection format needed
#for the ViP dataset
#PoseTrack2017 

import json
import os 
import glob

import numpy as np
from PIL import Image

root_dir = '/z/home/natlouis/data/posetrack17_data/posetrack_data/'
target_dir = '/z/home/natlouis/data/posetrack17_data/posetrack_data/annotations'

splits = ['train', 'val'] #no available test split?

#To gather statistics
track_ids = []
max_people_frm = 0
max_people_vid = 0

os.makedirs(target_dir, exist_ok=True)

for split in splits:
    split_dir = os.path.join(root_dir,'annotations/'+split)
    target_file = os.path.join(target_dir, split+'.json')

    json_files = sorted(glob.glob(os.path.join(split_dir,'*.json')))

    json_ann = []

    for annot in json_files:
        print(annot)
        with open(annot, 'r') as f:
            data = json.load(f)['annolist']

        frames = []
        unique_id_vid = []
        nframes = len(data)
        for anno in data:
            #expected keys: image, annorect, imgnum, is_labeled

            filename   = anno['image'][0]['name']
            anns       = anno['annorect'] 
            imgnum     = anno['imgnum'][0]
            is_labeled = anno['is_labeled'][0]

            #img_path   = os.path.join(root_dir,filename)
            img_path         = os.path.join('/home/natlouis/data/posetrack17_data',filename) #locally referenced JSON file 
            im = Image.open(img_path)
            width, height = im.size

            frame_id = int('1'+filename.split('/')[-2].split('_')[0]+str(imgnum).zfill(4))
            vid_id   = annot.split('/')[-1].split('.')[0]

            base_path = '/'.join(img_path.split('/')[:-1])
            
            frame      = {}
            objs       = []
            num_people = 0
            for ann in anns:
                #expected keys for each object: x1, y1, x2, y2, score, sale, track_id, annopoints

                #bbox around the head of the person
                x1 = ann['x1'][0]
                y1 = ann['y1'][0]
                x2 = ann['x2'][0]
                y2 = ann['y2'][0]

                score = ann['score'][0]
                scale = ann['scale'][0]
                track_id = ann['track_id'][0]
                obj_id   = str(frame_id).zfill(6)+str(track_id).zfill(2)
                class_id = 0

                annopoints = ann['annopoints']
                num_people += 1

                if len(annopoints) == 0: #body not annotated, must be occluded?
                    bbox_head = []
                    occ  = 1
                else:
                    xmin = float(np.clip(x1,0,min(x1,width)))
                    ymin = float(np.clip(y1,0,min(y1,height)))
                    xmax = float(np.clip(x2,0,width))
                    ymax = float(np.clip(y2,0,height))
                    bbox_head = [xmin, ymin, xmax, ymax]

                    occ  = 0

                keypoints = np.zeros((15,3), dtype=np.float32)
                if not occ:
                    for point in annopoints[0]['point']:
                        _id = int(point['id'][0])
                        x   = point['x'][0]
                        y   = point['y'][0]
                        vis = point['is_visible'][0]

                        keypoints[_id] = [x,y,vis]

                keypoints = keypoints.tolist()
                objs.append({'trackid':track_id, 'c':class_id, 'bbox_head':bbox_head, 'occ':occ,\
                             'keypoints':keypoints, 'image_id':frame_id,\
                             'id':obj_id})

                if track_id not in track_ids:
                    track_ids.append(track_id)
                if track_id not in unique_id_vid:
                    unique_id_vid.append(track_id)

            if len(unique_id_vid) > max_people_vid:
                max_people_vid = len(unique_id_vid)

            frame['objs'] = objs

            frame['img_path']  = img_path
            frame['nframes']   = nframes
            frame['frame_id']  = frame_id
            frame['vid_id']    = vid_id
            frame['is_labeled'] = is_labeled 
            frames.append(frame)

        json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

    print('Saving file to: {}'.format(target_file))
    #Write out to JSON file
    with open(target_file, 'w') as f:
        json.dump(json_ann, f)

print('Unique ids: {}, Max persons per frame: {}, Max persons per vid: {}'.format(len(track_ids), max_people_frm, max_people_vid))
