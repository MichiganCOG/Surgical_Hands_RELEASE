import os
import sys
import datetime
import yaml
import torch

import numpy                    as np

import torch.nn                 as nn
import torch.optim              as optim
import torch.utils.data         as Data

from tensorboardX                       import SummaryWriter

from parse_args                         import Parse
from models.models_import               import create_model_object
from datasets                           import data_loader 
from metrics                            import Metrics
from checkpoint                         import load_checkpoint

import pprint 

import wandb

def eval(**args):
    """
    Evaluate selected model 
    Args:
        seed       (Int):        Integer indicating set seed for random state
        save_dir   (String):     Top level directory to generate results folder
        model      (String):     Name of selected model 
        dataset    (String):     Name of selected dataset  
        exp        (String):     Name of experiment 
        load_type  (String):     Keyword indicator to evaluate the testing or validation set
        pretrained (Int/String): Int/String indicating loading of random, pretrained or saved weights
        
    Return:
        None
    """

    print("Experimental Setup: ")
    pprint.PrettyPrinter(indent=4).pprint(args)

    d          = datetime.datetime.today()
    date       = d.strftime('%Y%m%d-%H%M%S')
    result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],args['exp'],date)))
    log_dir    = os.path.join(result_dir, 'logs')
    save_dir   = os.path.join(result_dir, 'checkpoints')

    run_id = args['exp']
    use_wandb  = args.get('use_wandb', False)
    if not args['debug']:
        if use_wandb:
            wandb.init(project=args['dataset'], name=args['exp'], config=args, tags=args['tags'])

            #Replace result dir with wandb unique id, much easier to find checkpoints
            run_id = wandb.run.id

        if run_id:
            result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'], run_id)))
            log_dir    = os.path.join(result_dir, 'logs')
            save_dir   = os.path.join(result_dir, 'checkpoints')

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(log_dir,    exist_ok=True) 
        os.makedirs(save_dir,   exist_ok=True) 

        # Save copy of config file
        with open(os.path.join(result_dir, 'config.yaml'),'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

        # Tensorboard Element
        writer = SummaryWriter(log_dir)

    # Check if GPU is available (CUDA)
    num_gpus = args['num_gpus']
    device = torch.device("cuda:0" if num_gpus > 0 and torch.cuda.is_available() else "cpu")
    print('Using {}'.format(device.type))

    # Load Network
    model = create_model_object(**args).to(device)
    model_obj = model 

    if device.type == 'cuda' and num_gpus > 1:
        device_ids = list(range(num_gpus)) #number of GPUs specified
        model = nn.DataParallel(model, device_ids=device_ids)
        model_obj = model.module #Model from DataParallel object has to be accessed through module

        print('GPUs Device IDs: {}'.format(device_ids))

    # Load Data
    loader = data_loader(**args, model_obj=model_obj)

    if args['load_type'] == 'train_val':
        eval_loader = loader['valid']

    elif args['load_type'] == 'train':
        eval_loader = loader['train']

    elif args['load_type'] == 'test':
        eval_loader  = loader['test'] 

    else:
        sys.exit('load_type must be valid or test for eval, exiting')

    if isinstance(args['pretrained'], str):
        ckpt = load_checkpoint(args['pretrained'])

        ckpt_keys = list(ckpt.keys())
        if ckpt_keys[0].startswith('module.'): #if checkpoint weights are from DataParallel object
            for key in ckpt_keys:
                ckpt[key[7:]] = ckpt.pop(key)

        model_obj.load_state_dict(ckpt, strict=False)

    # Training Setup
    params     = [p for p in model.parameters() if p.requires_grad]

    acc_metric = Metrics(**args, result_dir=result_dir, ndata=len(eval_loader.dataset), logger=wandb if use_wandb else None, run_id=run_id)
    acc = 0.0

    # Setup Model To Evaluate 
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(eval_loader):
            x_input     = data['data']
            annotations = data['annots']

            if isinstance(x_input, torch.Tensor):
                outputs = model(x_input.to(device))
            else:
                for i, item in enumerate(x_input):
                    if isinstance(item, torch.Tensor):
                        x_input[i] = item.to(device)
                outputs = model(*x_input)

            if args['save_feat']:
                feats       = outputs['feat'].cpu().data
                gt_key_pts  = annotations['key_pts']
                bboxes      = annotations['bbox']
                obj_ids     = annotations['obj_ids']
                track_ids   = annotations['track_ids']
                vid_id      = annotations['vid_id']
                load_type   = annotations['load_type'][0]

                feat_dir = os.path.join(args['save_feat_dir'], args['model']+'-'+args['exp'], load_type)
                os.makedirs(feat_dir, exist_ok=True)

                for vid in set(vid_id):
                    idx = [i for i, item in enumerate(vid_id) if item == vid]

                    feat    = feats[idx]
                    key_pts = gt_key_pts[idx]
                    bbox    = bboxes[idx]
                    track   = track_ids[idx]
                    oid     = obj_ids[idx]

                    filename = os.path.join(feat_dir,vid+'.pkl') 
                    if os.path.exists(filename):
                        vid_data = torch.load(filename)
                        vid_data['feat']       = torch.cat((vid_data['feat'], feat))
                        vid_data['gt_key_pts'] = torch.cat((vid_data['gt_key_pts'], key_pts))
                        vid_data['bbox']       = torch.cat((vid_data['bbox'], bbox))
                        vid_data['track_id']   = torch.cat((vid_data['track_id'], track))
                        vid_data['object_ids'] = torch.cat((vid_data['object_ids'], oid))
                    else:
                        vid_data = {'feat':feat, 'gt_key_pts':key_pts, 'bbox':bbox, 'track_id':track, 'object_ids':oid}

                    torch.save(vid_data, filename)

                outputs = outputs['outputs']

            acc = acc_metric.get_accuracy(outputs, annotations)

            if step % 100 == 0:
                print('Step: {}/{} | {} acc: {:.4f}'.format(step, len(eval_loader), args['load_type'], acc))

    print('Accuracy of the network on the {} set: {:.3f} %\n'.format(args['load_type'], 100.*acc))

    if not args['debug']:
        if use_wandb:
            wandb.log({'val accuracy':100.*acc})

        writer.add_scalar(args['dataset']+'/'+args['model']+'/'+args['load_type']+'_accuracy', 100.*acc)
        # Close Tensorboard Element
        writer.close()

if __name__ == '__main__':

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    eval(**args)
