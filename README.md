# [[Temporally Guided Articulated Hand Pose Tracking in Surgical Videos](https://arxiv.org/abs/2101.04281)]

This source code was built using [ViP: Video Platform for PyTorch](https://github.com/MichiganCOG/ViP) some documentation and instructions can be found [here](https://github.com/MichiganCOG/ViP/wiki).

### Requirements
- Ubuntu 16.04+
- Python 3.6+
- PyTorch v1.0+
- CUDA 10 or 11

Recommended installation with VirtualEnvWrapper and [requirements.txt](https://github.com/MichiganCOG/Surgical_Hands_RELEASE/blob/main/requirements.txt)

(Optional) Supports [Weights & Biases](https://github.com/wandb/client) for logging, use `--use_wandb=True` in YAML file (--debug must also be set to 0).

## Datasets
 - **Mixed Hands** Dataset used for image pretraining are the Manual and Synthetic hand datasets (hence Mixed Hands) from the [CMU Panoptic Dataset](http://domedb.perception.cs.cmu.edu/handdb.html)
	- Extract to `$ROOT/data` directory and configure using `scripts/gen_json_mixed_hands.py`
 - **Surgical Hands** Our newly collected dataset that contains videos of surgical procedures accompanied with bounding box, pose, and tracking annotations. 
 	 - Download the following and extract to `$ROOT/data` directory
	  	- [Surgical Hands dataset](https://drive.google.com/file/d/1l5_4rlZLvOim34uHCKic4GUXvXfjDN_9/view?usp=sharing)
	  	- [Hand Detections](https://drive.google.com/file/d/1dWhZF595ixS-XBIeawaS3mY01yfsE_BO/view?usp=sharing)
	 - Configure using `scripts/gen_json_surgical_hands_folds_n.py` (for ground truth) and `scripts/gen_json_surgical_hands_dets_folds_n.py` (for detections) into formats needed for code base
	 - All experiments are done using k-fold cross validation and each data is split accordingly.

## Weights
- Download and extract to `$ROOT/weights` directory
	- ResNet152 ImageNet [weights](https://drive.google.com/file/d/14u4TYEpu6d6Eh4PsIOjeTYiMfc4nXAMe/view?usp=sharing)
	- Pretrained [weights](https://drive.google.com/drive/folders/1upSSUr4c2_SMmpzfQumoevNYhpig0UuW?usp=sharing) on Mixed Hands image dataset
	- Baseline [weights](https://drive.google.com/drive/folders/1skZGRnX_6SNiB-DROgg1RYBy4qI0BURB?usp=sharing) (trained on Surgical Hands)
	- Our model [weights](https://drive.google.com/drive/folders/1zCnU8drwr1Mzy4rOmaP3mQ-jSXiYOALN?usp=sharing) (trained on Surgical Hands)
	- As mentioned above, all experiments are done using k-fold cross validation. So there are k sets of weights for each model. A single set of weights can be trained on all data

## Training and Evaluation
### Pre-train on larger image dataset
`python train.py --cfg_file ./cfgs/config_hand_resnet.yaml --dataset Mixed_Hands --acc_metric PCK_FlowTrack --json_path ./data/hand_labels_mixed --model FlowTrack --epoch 75 --lr 1e-4 --batch_size 16 --milestones 40,60`

### Finetune on our (Surgical Hands) dataset
- (Baseline) `python train.py --cfg_file ./cfgs/config_train_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_fold$NUM --pretrained ./weights/Mixed_Hands/Mixed_Hands_best_model.pkl --tags folda$NUM`

- (Our model) `python train.py --cfg_file ./cfgs/config_train_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_fold$NUM --min_temporal_dist 3 --tags folda$NUM`

 
### Evaluation
For evaluation, we modify the [Poseval Evaluation repository](https://github.com/leonid-pishchulin/poseval) for hands instead of human pose (amongst other threshold and validation changes). All code is contained within [poseval\_hand](https://github.com/MichiganCOG/Surgical_Hands_RELEASE/tree/main/poseval_hand).
 
- (Baseline) `python eval.py --cfg_file cfgs/config_eval_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_folda$NUM --tags folda$NUM --pretrained ./weights/Surgical_Hands/FlowTrack/folda$NUM.pkl`

- (Our model) `python eval_cycle.py --cfg_file cfgs/config_eval_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_folda$NUM --tags folda$NUM --pretrained ./weights/Surgical_Hands_v2/FlowTrack_r_gt_v5_linear/folda$NUM.pkl`

- (Our model - detections) `python eval_cycle.py --cfg_file cfgs/config_eval_surgical_hands.yaml --dataset Hand_Dets --json_path ./data/pub_surgical_dets/annotations_folda$NUM --tags folda$NUM --pretrained ./weights/Surgical_Hands_v2/FlowTrack_r_gt_v5_linear/folda$NUM.pkl --det_threshold=0.1 --sc=2.75`

### Visualization
- (Baseline) `python eval.py --cfg_file cfgs/config_eval_surgical_hands_baseline.yaml --json_path ./data/pub_surgical/annotations_folda$NUM --tags folda$NUM --pretrained ./weights/Surgical_Hands/FlowTrack/folda$NUM.pkl --acc_metric Save_Video_Keypoints`

- (Our model) `python eval_cycle.py --cfg_file cfgs/config_eval_surgical_hands.yaml --json_path ./data/pub_surgical/annotations_folda$NUM --tags folda$NUM --pretrained ./weights/Surgical_Hands_v2/FlowTrack_r_gt_v5_linear/folda$NUM.pkl --acc_metric Save_Video_Keypoints`

If you find this data useful, please consider citing:

```
@article{louis2022temporally,
  title={Temporally guided articulated hand pose tracking in surgical videos},
  author={Louis, Nathan and Zhou, Luowei and Yule, Steven J and Dias, Roger D and Manojlovich, Milisa and Pagani, Francis D and Likosky, Donald S and Corso, Jason J},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--9},
  year={2022},
  publisher={Springer}
}
```

### Acknowledgements
This project was supported by grant number 1R01HL146619-01A1 from the National Institutes of Health and the Aikens Innovation Academy. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health or the Aikens Innovation Academy.  Support for The Michigan Society of Thoracic and Cardiovascular Surgeons Quality Collaborative (MSTCVS-QC) is provided by Blue Cross and Blue Shield of Michigan and Blue Care Network (BCBSM) as part of the BCBSM Value Partnerships program. Although Blue Cross Blue Shield of Michigan and MSTCVS-QC work collaboratively, the opinions, beliefs, and viewpoints expressed by the author do not necessarily reflect the opinions, beliefs, and viewpoints of BCBSM or any of its employees.
