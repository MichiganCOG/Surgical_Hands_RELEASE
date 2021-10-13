#Compile all JSON files and convert to the Detection format needed
#for the ViP dataset

import json
import os 
import glob

import numpy as np
from PIL import Image

root_dir = '/z/home/natlouis/data/posetrack_data'
#target_dir = '/z/home/natlouis/ViP/data_local_ref/posetrack_data/annotations' #annotations will be on local node instead of on /z
target_dir = '/z/home/natlouis/data/posetrack_data/annotations'

#splits = ['train', 'val', 'test']
splits = ['test']

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
        with open(annot, 'r') as f:
            data = json.load(f)

        imgs = data['images']
        cats = data['categories']

        img_path         = os.path.join(root_dir,imgs[0]['file_name'])
        im = Image.open(img_path)
        width, height = im.size

        frames = []
        unique_id_vid = []
        for img in imgs:
            frame = {}

            has_no_densepose = img['has_no_densepose'] 
            is_labeled       = img['is_labeled']
            img_path         = os.path.join(root_dir,img['file_name'])
            #img_path         = os.path.join('/home/natlouis/data/posetrack_data',img['file_name']) #locally referenced JSON file
            nframes          = img['nframes']
            frame_id         = img['frame_id']
            vid_id           = img['vid_id']

            base_path = '/'.join(img_path.split('/')[:-1])
            
            ignore_x = []
            ignore_y = []

            #Persons in these regions are not annotated
            if 'ignore_regions_x' in img or 'ignore_regions_y' in img:
                ignore_x = img['ignore_regions_x']
                ignore_y = img['ignore_regions_y']

            objs       = []
            if split != 'test': #Image contains pose annotations. Test set is not annotated
                anns = data['annotations']

                num_people = 0
                for ann in anns:
                    ann_id = ann['image_id']

                    #Find all annotations belonging to frame
                    if ann_id != frame_id:
                        continue 

                    num_people += 1
                    track_id   = ann['track_id']
                    keypoints  = ann['keypoints']
                    class_id   = ann['category_id']
                    obj_id     = ann['id']

                    x,y,w,h = ann['bbox_head']

                    xmin = float(np.clip(x,0,x))
                    ymin = float(np.clip(y,0,y))
                    xmax = float(np.clip(xmin+w,0,width))
                    ymax = float(np.clip(ymin+h,0,height))
                    bbox_head = [xmin, ymin, xmax, ymax]

                    if 'bbox' not in ann: #body not annotated, must be occluded?
                        bbox = []
                        occ  = 1
                    else:
                        x,y,w,h = ann['bbox']

                        xmin = np.clip(x,0,min(x,width))
                        ymin = np.clip(y,0,min(y,height))
                        xmax = np.clip(xmin+w,0,width)
                        ymax = np.clip(ymin+h,0,height)
                        bbox = [xmin, ymin, xmax, ymax]

                        occ  = 0

                    objs.append({'trackid':track_id, 'c':class_id, 'bbox':bbox, 'occ':occ,\
                                 'bbox_head':bbox_head, 'keypoints':keypoints, 'image_id':ann_id,\
                                 'id':obj_id, 'ignore_x':ignore_x, 'ignore_y':ignore_y})

                    if track_id not in track_ids:
                        track_ids.append(track_id)
                    if track_id not in unique_id_vid:
                        unique_id_vid.append(track_id)

                if num_people > max_people_frm:
                    max_people_frm = num_people
                if len(unique_id_vid) > max_people_vid:
                    max_people_vid = len(unique_id_vid)

            frame['objs'] = objs

            frame['img_path']  = img_path
            frame['nframes']   = nframes
            frame['frame_id']  = frame_id
            frame['vid_id']    = vid_id
            frame['is_labeled'] = is_labeled 
            frame['categories'] = data['categories']
            frames.append(frame)
        json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

    print('Saving file to: {}'.format(target_file))
    #Write out to CSV file
    with open(target_file, 'w') as f:
        json.dump(json_ann, f)

print('Unique ids: {}, Max persons per frame: {}, Max persons per vid: {}'.format(len(track_ids), max_people_frm, max_people_vid))
