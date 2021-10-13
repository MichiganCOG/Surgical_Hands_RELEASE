import argparse
import yaml
import torch

class Parse():

    def __init__(self):
        """
        Override config args with command-line args
        """
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--cfg_file', type=str, default='config_default_example.yaml', help='Configuration file with experiment parameters')

        #Command-line arguments will override any config file arguments
        parser.add_argument('--rerun',             type=int, help='Number of trials to repeat an experiment')
        parser.add_argument('--dataset',           type=str, help='Name of dataset')
        parser.add_argument('--batch_size',        type=int, help='Numbers of videos in a mini-batch (per GPU)')
        parser.add_argument('--pseudo_batch_loop', type=int, help='Number of loops for mini-batch')
        parser.add_argument('--num_gpus',          type=int, help='Number of GPUs to use, default: -1 (all available GPUs). 0 (use CPU), >1 (number of GPUs to use)')
        parser.add_argument('--num_workers',       type=int, help='Number of subprocesses for dataloading')
        parser.add_argument('--load_type',         type=str, help='Environment selection, to include only training/training and validation/testing dataset (train, train_val, test)')
        parser.add_argument('--model',             type=str, help='Name of model to be loaded')
        parser.add_argument('--labels',            type=int, help='Number of total classes in the dataset')

        parser.add_argument('--loss_type',    type=str,   help='Loss function')
        parser.add_argument('--acc_metric',   type=str,   help='Accuracy metric')
        parser.add_argument('--opt',          type=str,   help='Name of optimizer')
        parser.add_argument('--lr',           type=float, help='Learning rate')
        parser.add_argument('--momentum',     type=float, help='Momentum value in optimizer')
        parser.add_argument('--weight_decay', type=float, help='Weight decay')
        parser.add_argument('--milestones',   type=str, help='Epoch values to change learning rate')
        parser.add_argument('--gamma',        type=float, help='Multiplier with which to change learning rate')
        parser.add_argument('--epoch',        type=int,   help='Total number of epochs')

        parser.add_argument('--json_path',    type=str, help='Path to train and test json files')
        parser.add_argument('--save_dir',     type=str, help='Path to results directory')
        parser.add_argument('--exp',          type=str, help='Experiment name')
        parser.add_argument('--preprocess',   type=str, help='Name of the preprocessing method to load')
        parser.add_argument('--pretrained',   type=str, help='Load pretrained network or continue training (0 to randomly init weights, 1 to load default weights, str(path.pkl) to load checkpoint weights')
        parser.add_argument('--subtract_mean',type=str, help='Subtract mean (R,G,B) from all frames during preprocessing')
        parser.add_argument('--resize_shape', type=int, nargs=2,  help='(Height, Width) to resize original data')
        parser.add_argument('--final_shape',  type=int, nargs=2,  help='(Height, Width) of input to be given to CNN')
        parser.add_argument('--clip_length',  type=int, help='Number of frames within a clip')
        parser.add_argument('--clip_offset',  type=int, help='Frame offset between beginning of video and clip (1st clip only)')
        parser.add_argument('--random_offset',type=int, help='Randomly select clip_length number of frames from the video')
        parser.add_argument('--clip_stride',  type=int, help='Frame offset between successive frames')
        parser.add_argument('--crop_shape',   type=int, nargs=2,  help='(Height, Width) of frame') 
        parser.add_argument('--crop_type',    type=str, help='Type of cropping operation (Random, Center and None)')
        parser.add_argument('--num_clips',    type=int, help='Number clips to be generated from a video (<0: uniform sampling, 0: Divide entire video into clips, >0: Defines number of clips)')
        parser.add_argument('--scale',        type=float, nargs=2, help='[min scale, max scale] amounts to randomly scale videos for augmentation purposes. scale >1 zooms in and scale <1 zooms out.')
        parser.add_argument('--sc',           type=float, help='Sole purpose of parameter searching for PoseTrack18')
        parser.add_argument('--conf_threshold', type=float, help='Threshold for keypoint confidence, when outputting JSON file')
        parser.add_argument('--hm_to_layer',  type=int, help='Append heatmap prior to this resnet layer [1,2,3,4]')
        parser.add_argument('--bm_threshold', type=float, help='Threshold to create binary mask from heatmap prior')
        parser.add_argument('--det_threshold', type=float, help='Threshold for confidence of object detections')
        parser.add_argument('--ignore_reg_threshold', type=float, help='Threshold to filter detections overlapping with annotated ignore regions')
        parser.add_argument('--nms_threshold', type=float, help='Threshold for NMS, if applicable')
        parser.add_argument('--min_temporal_dist', type=int, help='Minimum temporal distance between prior time step and current time step')
        parser.add_argument('--match_strategy', type=str, help='Pose matching during tracking')
        parser.add_argument('--spa_con_thresh', type=float, help='Threshold for spatial consistency')
        parser.add_argument('--l2_margin', type=float, help='Margin for l2 matching strategy')
        parser.add_argument('--cont_acc_margin', type=float, help='Margin for gcn matching strategy')
        parser.add_argument('--last_T_frames', type=int, help='Match with last T frames')
        parser.add_argument('--gcn_checkpoint', type=str, help='saved gcn model checkpoint, for match_strategy=gcn')
        parser.add_argument('--gcn_vis_embed', type=int, help='Enable joint visualization embedding for GCN')
        parser.add_argument('--gcn_feat_dir', type=str, help='Directory of features to use for GCN training and eval')

        #hyper-parameters for recursive FlowTrack models
        parser.add_argument('--prior_threshold',      type=float, help='Threshold to use prediction_{t-1} or ground truth_{t-1} as prior during training')
        parser.add_argument('--min_gauss_peak_train', type=float, help='Minimum confidence value to use as target for gaussian peaks, during training')
        parser.add_argument('--min_gauss_peak_eval',  type=float, help='Minimum confidence value to use as target for gaussian peaks, during evaluation')
        parser.add_argument('--debug',           type=int, help='Run an experiment but do not save any data or create any folders')
        parser.add_argument('--save_feat',       action='store_true', help='Only during evaluation, save selected features from a network')
        parser.add_argument('--out_feat',        action='store_true', help='Output features during evaluation, used in GCN-Vis')
        parser.add_argument('--save_feat_dir',   type=str, help='Location to save features')
        parser.add_argument('--viz',     action='store_true', help='Visualization flag. Insert any matplotlib figures, downloading images, or pop-up windows under this condition')
        parser.add_argument('--seed',    type=int, help='Seed for reproducibility')
        parser.add_argument('--resume',  type=int, help='Flag to resume training or switch to alternate objective after loading')

        parser.add_argument('--tags', type=str, help='comma separated tags used for logging purposes')

        # Default dict, anything not present is required to exist as an argument or in yaml file
        self.defaults = dict(
            rerun                = 5,
            batch_size           = 1,
            pseudo_batch_loop    = 1,
            num_gpus             = -1,
            num_workers          = 1,
            acc_metric           = None,
            opt                  = 'sgd',
            lr                   = 0.001,
            momentum             = 0.9,
            weight_decay         = 0.0005,
            milestones           = [5],
            gamma                = 0.1,
            epoch                = 10,
            save_dir             = './results',
            exp                  = 'exp',
            preprocess           = 'default',
            pretrained           = 0,
            subtract_mean        = '',
            clip_offset          = 0,
            random_offset        = 0,
            clip_stride          = 0,
            crop_type            = None,
            num_clips            = 1,
            debug                = 0,
            viz                  = 0,
            seed                 = 0,
            scale                = [1,1],
            sc                   = 1.15,
            conf_threshold       = 0.0,
            hm_to_layer          = 1,
            bm_threshold         = -1,
            det_threshold        = 0.0,
            ignore_reg_threshold = 0.05,
            nms_threshold        = 0.9,
            min_temporal_dist    = 4,
            match_strategy       = 'sc',
            spa_con_thresh       = 0.5,
            l2_margin            = 1000.0,
            cont_acc_margin      = 0.2,
            last_T_frames        = 1,
            gcn_checkpoint       = None,
            gcn_vis_embed        = 0,
            gcn_feat_dir         = './',
            prior_threshold      = 0.5,
            min_gauss_peak_train = 0.0,
            min_gauss_peak_eval  = 0.0,
            save_feat            = 0,
            out_feat             = 0,
            save_feat_dir        = './features',
            resume               = 0,
            tags                 = '')                       

        #Dictionary of the command-line arguments passed
        self.cmd_args = vars(parser.parse_args()) 

        config_file = self.cmd_args['cfg_file']
        with open(config_file, 'r') as f:
            self.cfg_args = yaml.safe_load(f) #config file arguments

    def get_args(self):
        yaml_keys = self.cfg_args.keys() 

        # If pretrained is the string 0 or 1, set it to int, otherwise leave the path as a string
        if 'pretrained' in yaml_keys:
            v = self.cfg_args['pretrained']
            if v=='0' or v=='1':
                self.cfg_args['pretrained'] = int(v)


        for (k,v) in self.cmd_args.items():
            if (k == 'pretrained'):
                if v=='0' or v=='1':
                    v = int(v)


            if v is not None:
                if k == 'milestones':
                    v = [int(e) for e in v.replace(',',' ').split(' ')]

                self.cfg_args[k] = v

                if k == 'tags':
                    v = v.split(',')
                    self.cfg_args[k] = v
            else:
                if k not in yaml_keys:
                    self.cfg_args[k] = self.defaults[k]


        # Force clip_stride to be >= 1 when extracting clips from a video
        # This represents the # of frames between successive clips 
        if self.cfg_args['clip_stride'] < 1:
            self.cfg_args['clip_stride'] = 1

	#Use all available GPUs if num_gpus = -1
        #Else select the minimum between available GPUS and requested GPUs
        num_gpus = torch.cuda.device_count() if self.cfg_args['num_gpus'] == -1 else min(torch.cuda.device_count(), self.cfg_args['num_gpus'])
        self.cfg_args['num_gpus'] = num_gpus 

        return self.cfg_args
