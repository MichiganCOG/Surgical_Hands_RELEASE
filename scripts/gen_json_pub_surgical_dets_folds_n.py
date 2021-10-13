#Reformat JSON file for the ViP dataset
#Separate data into N folds, for leave-one-out cross validatioiin
#Uses object detections

import json
import os 
import glob

import numpy as np
import torch
from PIL import Image

import subprocess

source_obj_dets  = 'data/hand_detections/'

source_json_file = 'data/surgical_hands_release/annotations.json'
source_res_dir   = 'data/surgical_hands_release/images'

target_json_dir = 'data/pub_surgical_dets/'

#certain factors (poseval) requires numeric video names
vid2idx = {'fTFTk_q8dh0':'000001'
,'e45vsP9CM2c':'000002'
,'19P6qybcoUE':'000003'
,'eb984ec0cf5':'000004'
,'5uKdXnreV1s':'000005'
,'alzo8uZzhpk':'000006'
,'v_AngiPw1wc':'000007'
,'0318dd3e7e3':'000008'
,'09116df3238':'000009'
,'6iSraYGcHBk':'000010'
,'B5K_QYc_Y2o':'000011'
,'QokL8kNka9g':'000012'
,'K4WxM7z1PKo':'000013'
,'GkfnPAgdaUI':'000014'
,'a8c1424ebcd':'000015'
,'8jXjg4uaES0':'000016'
,'68e2d14b311':'000017'
,'HWpy-ZzKSQE':'000018'
,'a7d419fca90':'000019'
,'01560685f81':'000020'
,'3d384b2969e':'000021'
,'5449be235ad':'000022'
,'3745b0bac16':'000023'
,'TsG-g925r8s':'000024'
,'G2lrwv61Mss':'000025'
,'9NM1_cC5PfI':'000026'
,'ZjjgFgisJwI':'000027'
,'9DhftZeReiI':'000028'
}

#Create all folds from src videos
folds = {}
for idx, vid_name in enumerate(vid2idx.keys()):
    fold_name = 'folda'+str(idx)

    folds[vid_name] = {}
    folds[vid_name]['target_dir']  = os.path.join(target_json_dir, 'annotations_'+fold_name)

for fold,val in folds.items(): #each cross-validation fold
    target_dir  = val['target_dir']
    train_vids  = list(folds.keys()); train_vids.remove(fold)
    test_vids   = [fold]

    joint_names = ['wrist', 'thumb_k', 'thumb_b', 'thumb_m', 'thumb_t', \
                    'index_k', 'index_b', 'index_m', 'index_t', \
                    'middle_k', 'middle_b', 'middle_m', 'middle_t', \
                    'ring_k', 'ring_b', 'ring_m', 'ring_t', \
                    'pinky_k', 'pinky_b', 'pinky_m', 'pinky_t']

    neighbor_link = [[0,1], [1,2], [2,3], [3,4],
                    [0,5], [5,6], [6,7], [7,8],
                    [0,9], [9,10], [10,11], [11,12],
                    [0,13], [13,14],[14,15], [15,16],
                    [0,17], [17,18], [18,19], [19,20]]

    categories = {'supercategory':'hand',
                   'id':2,
                   'name':'hand',
                   'keypoints':joint_names,
                   'skeleton':neighbor_link}

    splits = ['train', 'test']

    joint_names = ['wrist', 'thumb_k', 'thumb_b', 'thumb_m', 'thumb_t', \
		    'index_k', 'index_b', 'index_m', 'index_t', \
		    'middle_k', 'middle_b', 'middle_m', 'middle_t', \
		    'ring_k', 'ring_b', 'ring_m', 'ring_t', \
		    'pinky_k', 'pinky_b', 'pinky_m', 'pinky_t']

    neighbor_link = [[0,1], [1,2], [2,3], [3,4],
		    [0,5], [5,6], [6,7], [7,8],
		    [0,9], [9,10], [10,11], [11,12],
		    [0,13], [13,14],[14,15], [15,16],
		    [0,17], [17,18], [18,19], [19,20]]

    categories = {'supercategory':'hand',
		   'id':2,
		   'name':'hand',
		   'keypoints':joint_names,
		   'skeleton':neighbor_link}

    os.makedirs(target_dir, exist_ok=True)

    with open(source_json_file, 'r') as f:
        data = json.load(f)

    num_clips = 0
    T = '4' #surg
    for split in splits:
        print('\nsplit: {}'.format(split))
        target_file = os.path.join(target_dir, split+'.json')
        
        json_ann = []
        for vid in sorted(data.keys()):

            _vid = vid.split('_')

            #vid names are overly complicated, contains underscores, multiple parts, etc
            if len(_vid[-1].zfill(2)) == 2:
                vid_src = '_'.join(vid.split('_')[:-3])
                ss = _vid[-3]
            else:
                vid_src = '_'.join(vid.split('_')[:-2])
                ss = _vid[-2]

            #if vid == 'ZjjgFgisJwI_000050_000135_1':
            #    import pdb; pdb.set_trace()

            if split == 'train' and vid_src not in train_vids:
                continue
            if split == 'test' and vid_src not in test_vids:
                continue

            num_clips += 1
            imgs = data[vid]['images']      
            anns = data[vid]['annotations']

            file_name  = imgs[0]['file_name']
            base_path = os.path.join(source_res_dir, vid)
            #Create symbolic link with new sequence name
            seq_name = '_'.join((vid,'surg',split))
            
            print('seq: {}'.format(vid))

            det_file = os.path.join(source_obj_dets, vid+'.pkl')
            det_data = torch.load(det_file)

            dets    = det_data['dets'] #adding a scaling to offset from box annotation creation
            scores  = det_data['scores']
            classes = det_data['class']
            paths   = det_data['frame_paths']

            det_width  = 640
            det_height = 360
            frames = []
            width, height = None, None
            for item in imgs:
                file_name  = item['file_name']
                is_labeled = item['is_labeled']
                vid_id     = item['video_dir']
                iid        = item['id']

                if not is_labeled: #Only provided labeled frames
                    continue 

                if width == None or height == None:
                    #Get image dimensions
                    img_path = os.path.join(base_path, file_name)
                    im = Image.open(img_path)
                    width, height = im.size

                    dets[:,0] *= (width/det_width)
                    dets[:,1] *= (height/det_height)
                    dets[:,2] *= (width/det_width)
                    dets[:,3] *= (height/det_height)

                frame_num = iid.split('_')[-1]
                seq_id = vid2idx[vid_src]
                frame_id = int(T+seq_id[3:]+ss+frame_num[2:])

                img_path    = os.path.join(base_path, file_name)
                #cats = data['categories']
                #item_anns = [ann for ann in anns if ann['image_id'] == iid]

                indices = [i for i,j in enumerate(paths) if '/'.join((j.split('/')[-2:])) == '/'.join((img_path.split('/')[-2:]))]
                if len(indices) == 0: #no detections
                    pass
                else:
                    det_objs = dets[indices] #[x1, y1, x2, y2]
                    scr_objs = scores[indices]
                    cls_objs = classes[indices]

                objs = []
                frame = {}
                for idx, (obj, cls_score, cls_pred) in enumerate(zip(det_objs, scr_objs, cls_objs)):
                    x1,y1,x2,y2 = obj.tolist()

                    tid  = -1
                    kpts = [-1]*(21*3)
                    class_id = cls_pred.item() #0: left, 1: right
                    obj_id = '_'.join((vid,frame_num, str(idx).zfill(2)))

                    bbox = [x1, y1, x2, y2]
                    occ = np.array(kpts).reshape((21,3))[:,-1] == 0

                    xmin, ymin, xmax, ymax = bbox

                    objs.append({'trackid':tid, 'c':class_id, 'bbox':bbox, 'occ':occ.tolist(), \
                                'confidence':0, 'cls_score':cls_score.item(), 'cls_pred':cls_pred.item(), \
                                 'hand_pts':kpts, 'image_id':iid, 'id':obj_id})

                frame['objs'] = objs

                frame['img_path']  = img_path
                frame['frame_id']  = frame_id
                frame['vid_id']    = seq_name
                frame['is_labeled'] = is_labeled 
                #frame['categories'] = data['categories']
                frames.append(frame)

            json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

        print('Saving file to: {}'.format(target_file))
        with open(target_file, 'w') as f:
            json.dump(json_ann, f)
    print('{} clips in fold {}'.format(num_clips, fold))

