# [[Temporally Guided Articulated Hand Pose Tracking in Surgical Videos](https://arxiv.org/abs/2101.04281)]

This source code was built using [ViP: Video Platform for PyTorch](https://github.com/MichiganCOG/ViP) some documentation and instructions can be found [here](https://github.com/MichiganCOG/ViP/wiki).

### Requirements
- Ubuntu 16.04 or later
- PyTorch v1.0 or later
- CUDA 10 or later

## Dataset
 - Download the following and extract to `$ROOT/data` directory
	 - [Surgical Hands dataset](https://drive.google.com/file/d/1l5_4rlZLvOim34uHCKic4GUXvXfjDN_9/view?usp=sharing)
	 - [Hand Detections](https://drive.google.com/file/d/1dWhZF595ixS-XBIeawaS3mY01yfsE_BO/view?usp=sharing)
 - Configure using `scripts/gen_json_surgical_hands_folds_n.py` (for ground truth) and `scripts/gen_json_surgical_dets_hands_folds_n.py` (for detections) into formats needed for code base
	 - All experiments are done using k-fold cross validation and each data is split accordingly.

## Weights
- Download and extract to `$ROOT/weights` directory
	- Pretrained [weights](https://drive.google.com/drive/folders/1upSSUr4c2_SMmpzfQumoevNYhpig0UuW?usp=sharing) on Mixed Hands image dataset
	- Baseline [weights](https://drive.google.com/drive/folders/1skZGRnX_6SNiB-DROgg1RYBy4qI0BURB?usp=sharing) (trained on Surgical Hands)
	- Our model [weights](https://drive.google.com/drive/folders/1zCnU8drwr1Mzy4rOmaP3mQ-jSXiYOALN?usp=sharing) (trained on Surgical Hands)
	- As mentioned above, all experiments are done using k-fold cross validation. So there are k sets of weights for each model. A single set of weights can be trained on all data

## Training and Evaluation
### Pre-train on larger image dataset
`python train.py --cfg_file ./cfgs/config_hand_resnet --dataset MixedHands --acc_metric PCK_FlowTrack --json_path TBD --model FlowTrack --epoch 75 --lr 1e-4 --batch_size 16 --milestones 40,60 `

### Finetune on our (Surgical Hands) dataset
- (Baseline) `python train.py --cfg_file ./cfgs/config_hand_resnet.yaml --dataset SurgicalHands --acc_metric PCK_FlowTrack --json_path ./data/pub_surgical/annotations_fold$NUM --model FlowTrack --epoch 20 --lr 8e-5 --batch_size 12 --milestones 5  --pretrained ./weights/Mixed_Hands/Mixed_Hands_best_model.pkl --tags folda$NUM`
- (Our model) `python train.py --cfg_file ./cfgs/config_hand_resnet.yaml --dataset SurgicalHands_v2 --acc_metric PCK_FlowTrack --json_path ./data/pub_surgical/annotations_fold$NUM --model FlowTrack_r_gt_v5_linear --epoch 20 --lr 8e-5 --batch_size 12 --milestones 20 --pretrained ./weights/Mixed_Hands/Mixed_Hands_best_model.pkl --min_temporal_dist 3 --prior_threshold 0.5 --min_gauss_peak_train 0.0 --tags folda$NUM`
- 
### Evaluation
- (Baseline) `python eval.py --cfg_file ./cfgs/config_hand_resnet.yaml --dataset Surgical_Hands --acc_metric Eval_PoseTrack18_det --model FlowTrack --match_strategy sc --spa_con_thres 0.2 --conf_threshold 0.43 --pretrained ./weights/Surgical_Hands/FlowTrack/folda$NUM.pkl --json_path ./data/pub_surgical/annotations_fold$NUM --tags folda$NUM `

- (Our model) `python eval_cycle.py --cfg_file ./cfgs/config_hand_resnet.yaml --dataset Surgical_Hands --acc_metric Eval_PoseTrack18_det --model FlowTrack_r_gt_v5_linear --match_strategy sc --spa_con_thres 0.2 --conf_threshold 0.43 --pretrained ./weights/Surgical_Hands_v2/FlowTrack_r_gt_v5_linear/folda$NUM.pkl --json_path ./data/pub_surgical/annotations_fold$NUM --tags folda$NUM`

### Visualization
- (Baseline) Same as evaluation command, except replace `--acc_metric Save_Video_Keypoints`
- (Our model) Same as evaluation command, except replace `--acc_metric Save_Video_Keypoints`

If you find this data useful, please consider citing:

`Louis, Nathan, Luowei Zhou, Steven J. Yule, Roger D. Dias, Milisa Manojlovich, Francis D. Pagani, Donald S. Likosky, and Jason J. Corso. "Temporally Guided Articulated Hand Pose Tracking in Surgical Videos." _arXiv preprint arXiv:2101.04281_ (2021).`
