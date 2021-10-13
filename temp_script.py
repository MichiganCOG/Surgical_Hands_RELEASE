import os
import glob
import subprocess

model = 'FlowTrack_r_gt_v5_linear'
dataset = 'Surgical_Hands_v2'
root_dir = '/z/home/natlouis/ViP/TMI_checkpoints/'+model 
target_dir = './weights/'+dataset+'/'+model

os.makedirs(target_dir, exist_ok=True)

source_files = sorted(glob.glob(os.path.join(root_dir, '*/checkpoints/'+dataset+'_best_model.pkl')))

for sf in source_files:
    fold = sf.split('/')[-3].split('_')[0]
    print(fold)
    
    subprocess.check_output(['cp', sf, os.path.join(target_dir, fold+'.pkl')])
