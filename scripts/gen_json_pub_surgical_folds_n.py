#Reformat JSON file for the ViP dataset
#Separate data into N folds, for leave-one-out cross validatioiin
#Uses Groundtruth annotations

import json
import os 
import glob

import numpy as np
from PIL import Image

import subprocess

source_json_file = 'data/surgical_hands_release/annotations.json'
source_res_dir   = 'data/surgical_hands_release/images'

target_json_dir = 'data/pub_surgical/'
target_poseval_dir = '/z/home/natlouis/Surgical_Hands/temp_dir/'

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
    folds[vid_name]['poseval_dir'] = os.path.join(target_poseval_dir, 'target_'+fold_name)

for fold,val in folds.items(): #each cross-validation fold
    target_dir  = val['target_dir']
    poseval_dir = val['poseval_dir']
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
    os.makedirs(poseval_dir, exist_ok=True)

    with open(source_json_file, 'r') as f:
        data = json.load(f)

    def make_poseval_json(target_dir, seq_name, poseval_json_data):
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, seq_name+'.json')

        poseval_json_data['categories'] = [categories]
        print('Saving file to: {}'.format(target_file))
        with open(target_file, 'w') as f:
            json.dump(poseval_json_data, f)

    num_clips = 0
    T = '4' #surg
    avg_bbox_area = 0
    avg_frame_area = 0
    num_objects = 0
    num_imgs = 0
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
            poseval_json_data = {'images':[], 'annotations':[]}

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

                frame_num = iid.split('_')[-1]
                seq_id = vid2idx[vid_src]
                frame_id = int(T+seq_id[3:]+ss+frame_num[2:])

                img_path    = os.path.join(base_path, file_name)
                #cats = data['categories']
                item_anns = [ann for ann in anns if ann['image_id'] == iid]

                objs = []
                poseval_objs = []
                frame = {}
                for obj in item_anns:
                    obj_id = obj['id']
                    tid    = obj['track_id']
                    cat    = obj['category_id']
                    bbox   = obj['bbox']
                    kpts   = obj['keypoints']

                    xmin, ymin, xmax, ymax = bbox

                    num_objects += 1
                    avg_bbox_area += ((xmax-xmin) * (ymax-ymin))

                    occ = np.array(kpts).reshape((21,3))[:,-1] == 0

                    objs.append({'trackid':tid, 'c':cat, 'bbox':bbox, 'occ':occ.tolist(), \
                                 'hand_pts':kpts, 'image_id':iid, 'id':obj_id})

                    kpts = np.array(kpts, dtype=np.float32).reshape((21,3)) #n/a: 0, occ: 1, vis: 2

                    kpts[...,-1] = np.clip(kpts[...,-1],0,1) #n/a: 0, occ: 1, vis: 1
                    #kpts[...,-1] = kpts[...,-1]*0.5 #n/a: 0, occ: 0.5, vis: 1

                    poseval_objs.append({'keypoints':kpts.reshape(-1).tolist(),
                                         'track_id':tid,
                                         'image_id':frame_id,
                                         'bbox':bbox,
                                         'bbox_head':bbox,
                                         'scores':[],
                                         'category_id':2,
                                         'id':str(frame_id)+str(tid).zfill(2)})

                if split == 'test':
                    poseval_json_data['images'].append({'has_no_densepose':True,
                        'is_labeled':is_labeled,
                        'file_name':img_path,
                        'nframes':len(imgs),
                        'frame_id':frame_id,
                        'vid_id':seq_name,
                        'id':frame_id})
                    
                    poseval_json_data['annotations'].extend(poseval_objs)

                frame['objs'] = objs

                frame['img_path']  = img_path
                frame['frame_id']  = frame_id
                frame['vid_id']    = seq_name
                frame['is_labeled'] = is_labeled 
                #frame['categories'] = data['categories']
                frames.append(frame)

                if is_labeled:
                    avg_frame_area += width * height
                    num_imgs += 1
            json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

            if split == 'test':
                make_poseval_json(poseval_dir, seq_name, poseval_json_data)

        print('Saving file to: {}'.format(target_file))
        with open(target_file, 'w') as f:
            json.dump(json_ann, f)

    avg_bbox_area /= num_objects
    avg_frame_area /= num_imgs
    print('{} objects'.format(num_objects))
    print('{} annotated frames'.format(num_imgs))
    print('Average bbox area: {:.2f} px, {:.2f}^2 px'.format(avg_bbox_area, np.sqrt(avg_bbox_area)))
    print('Average frame area: {:.2f} px, {:.2f}^2 px'.format(avg_frame_area, np.sqrt(avg_frame_area)))
    print('Percentage: {:.2f}'.format(avg_bbox_area/avg_frame_area)) 

    print('{} clips in fold {}'.format(num_clips, fold))

