import os
import sys 
import datetime
import yaml 
import torch

import numpy             as np
import torch.nn          as nn
import torch.optim       as optim

from torch.optim.lr_scheduler           import MultiStepLR
from tensorboardX                       import SummaryWriter

from parse_args                         import Parse
from models.models_import               import create_model_object
from datasets.loading_function          import data_loader 
from losses                             import Losses
from metrics                            import Metrics
from checkpoint                         import save_checkpoint, load_checkpoint

import pprint

import wandb

def train(**args):
    """
    Evaluate selected model 
    Args:
        rerun        (Int):        Integer indicating number of repetitions for the select experiment 
        seed         (Int):        Integer indicating set seed for random state
        save_dir     (String):     Top level directory to generate results folder
        model        (String):     Name of selected model 
        dataset      (String):     Name of selected dataset  
        exp          (String):     Name of experiment 
        debug        (Int):        Debug state to avoid saving variables 
        load_type    (String):     Keyword indicator to evaluate the testing or validation set
        pretrained   (Int/String): Int/String indicating loading of random, pretrained or saved weights
        opt          (String):     Int/String indicating loading of random, pretrained or saved weights
        lr           (Float):      Learning rate 
        momentum     (Float):      Momentum in optimizer 
        weight_decay (Float):      Weight_decay value 
        final_shape  ([Int, Int]): Shape of data when passed into network
        
    Return:
        None
    """

    print("Experimental Setup: ")
    pprint.PrettyPrinter(indent=4).pprint(args)

    for total_iteration in range(args['rerun']):

        # Generate Results Directory
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
        loader = data_loader(model_obj=model_obj, **args)

        if args['load_type'] == 'train':
            train_loader = loader['train']
            valid_loader = loader['train'] # Run accuracy on train data if only `train` selected

        elif args['load_type'] == 'train_val':
            train_loader = loader['train']
            valid_loader = loader['valid'] 

        else:
            sys.exit('Invalid environment selection for training, exiting')

        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['model'] == 'Hand':
            named_params = [(n,p) for (n,p) in model.named_parameters() if p.requires_grad]

            group0_params = []
            group0_bias_params = []
            group1_params = []
            group1_bias_params = []

            for n,p in named_params:
                if 'model1_0' in n:
                    if 'bias' in n:
                        group0_bias_params.append(p)
                    else:
                        group0_params.append(p)
                else:
                    if 'bias' in n:
                        group1_bias_params.append(p)
                    else:
                        group1_params.append(p)

            params = [{'params': group0_params}, {'params': group0_bias_params, 'lr': 2*args['lr']},
                      {'params': group1_params, 'lr': 4*args['lr']}, {'params': group1_bias_params, 'lr': 8*args['lr']}
                     ]
        elif args['model'] == 'FlowTrack_R' or \
             args['model'] == 'FlowTrack_R_GT': #Set the learning rate of layer1.conv1 to be higher than others
            names = [n for (n,p) in model.named_parameters() if p.requires_grad]

            named_params = [(n,p) for (n,p) in model.named_parameters() if p.requires_grad]

            group0_params = []
            group1_params = []

            for n,p in named_params:
                if n == 'layer1.0.conv1.weight':
                    group0_params.append(p)
                else:
                    group1_params.append(p)

            params = [{'params': group0_params}, {'params': group1_params, 'lr': 0.1*args['lr']}]

        if args['opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'], nesterov=True)

        elif args['opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['lr'], weight_decay=args['weight_decay'])
        
        else:
            sys.exit('Unsupported optimizer selected. Exiting')

        scheduler  = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])

        if isinstance(args['pretrained'], str):
            ckpt        = load_checkpoint(args['pretrained'])

            ckpt_keys = list(ckpt.keys())
            if ckpt_keys[0].startswith('module.'): #if checkpoint weights are from DataParallel object
                for key in ckpt_keys:
                    ckpt[key[7:]] = ckpt.pop(key)

            models_to_modify = ['FlowTrack_R', 'FlowTrack_R_V2', 'FlowTrack_R_GT', 'FlowTrack_R_GT_No_Aug', 'FlowTrack_R_GT_V2']
            if args['model'] in models_to_modify:
                idx = args['hm_to_layer']
                layers_to_remove = ['layer'+str(idx)+'.0.conv1.weight', 'layer'+str(idx)+'.0.downsample.0.weight']

                for l in layers_to_remove:
                    if ckpt[l].shape[1] != model_obj.state_dict()[l].shape[1]: #initialize last few channels with zeros, but leave remaining weights
                        del ckpt[l]

            model_obj.load_state_dict(ckpt, strict=False)

            if args['resume']:
                start_epoch = load_checkpoint(args['pretrained'], key_name='epoch') + 1

                optimizer.load_state_dict(load_checkpoint(args['pretrained'], key_name='optimizer'))

                for _ in range(start_epoch):
                    scheduler.step()

            else:
                start_epoch = 0

        else:
            start_epoch = 0

        model_loss = Losses(device=device, **args)

        best_val_acc = 0.0

        # Start: Training Loop
        print('Starting Schedulers lr: {}'.format(scheduler.get_last_lr()[0]))
        for epoch in range(start_epoch, args['epoch']):
            acc_metric = Metrics(**args, ndata=len(train_loader.dataset), logger=wandb if use_wandb else None)
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Setup Model To Train 
            model.train()

            if args['model'] == 'FlowTrack_r_gt_v5_linear':
                if num_gpus > 1:
                    model.module.update_epoch(epoch)
                else:
                    model.update_epoch(epoch)

            # Start: Epoch
            for step, data in enumerate(train_loader):
                if step% args['pseudo_batch_loop'] == 0:
                    loss = 0.0
                    running_batch = 0
                    optimizer.zero_grad()

                x_input       = data['data'] 
                annotations   = data['annots']

                if isinstance(x_input, torch.Tensor):
                    mini_batch_size = x_input.shape[0]
                    outputs = model(x_input.to(device))

                    assert args['final_shape']==list(x_input.size()[-2:]), "Input to model does not match final_shape argument"
                else: #Model takes several inputs in forward function 
                    mini_batch_size = x_input[0].shape[0] #Assuming the first element contains the true data input 
                    for i, item in enumerate(x_input):
                        if isinstance(item, torch.Tensor):
                            x_input[i] = item.to(device)
                    outputs = model(*x_input)
                
                loss    = model_loss.loss(outputs, annotations)
                loss    = loss * mini_batch_size 
                loss.backward()

                running_loss  += loss.item()
                running_batch += mini_batch_size

                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()

                if not args['debug']:

                    # Add Learning Rate Element
                    for param_group in optimizer.param_groups:
                        if use_wandb:
                            wandb.log({'lr':param_group['lr'],'train loss':loss.item()/mini_batch_size})
                        writer.add_scalar(args['dataset']+'/'+args['model']+'/learning_rate', param_group['lr'], epoch*len(train_loader) + step)

                    # Add Training Loss Element
                    writer.add_scalar(args['dataset']+'/'+args['model']+'/minibatch_loss', loss.item()/mini_batch_size, epoch*len(train_loader) + step)

                    #Compute and Log Training Accuracy
                    with torch.no_grad():
                        # Add Training Accuracy Element
                        acc = acc_metric.get_accuracy(outputs, annotations)
                    if use_wandb:
                        wandb.log({'train accuracy':acc.item()})

                if ((epoch*len(train_loader) + step+1) % 100 == 0):
                    print('Epoch: {}/{}, step: {}/{} | train loss: {:.5f}'.format(epoch, args['epoch'], step+1, len(train_loader), running_loss/float(step+1)/mini_batch_size))

                if (epoch * len(train_loader) + (step+1)) % args['pseudo_batch_loop'] == 0 and step > 0:
                    # Apply large mini-batch normalization
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad *= 1./float(running_batch)

                    # Apply gradient clipping
                    if ("grad_max_norm" in args) and float(args['grad_max_norm'] > 0):
                        nn.utils.clip_grad_norm_(model.parameters(),float(args['grad_max_norm']))

                    optimizer.step()
                    running_batch = 0

            scheduler.step()
            print('Schedulers lr: {}'.format(scheduler.get_last_lr()[0]))

            ''' #For now, avoid saving every checkpoint
            if not args['debug']:
                # Save Current Model
                save_path = os.path.join(save_dir, args['dataset']+'_epoch'+str(epoch)+'.pkl')
                save_checkpoint(epoch, step, model, optimizer, save_path)
                print('Saved checkpoint to: {}'.format(save_path))
            '''
   
            prior_track_models = ['FlowTrack_r_gt_v5_no_max','FlowTrack_r_gt_v5_linear']
            if not args['debug'] and args['model'] in prior_track_models:
                if use_wandb:
                    wandb.log({'epoch':epoch, 'pred_to_prior': model.use_pred/model.total_priors})
                print('total_priors: {}, use_gt: {}, use_pred: {}'.format(model.total_priors, model.use_gt, model.use_pred))

            ## START FOR: Validation Accuracy
            running_acc  = []
            running_acc = valid(valid_loader, running_acc, model, model_loss, device)

            if args['model'] in prior_track_models:
                model.reset_vals() #Reset the values for tracking usage of predictions priors or gt priors

            if not args['debug']:
                if use_wandb:
                    wandb.log({'epoch':epoch, 'val accuracy':running_acc[-1]})

                writer.add_scalar(args['dataset']+'/'+args['model']+'/validation_accuracy', 100.*running_acc[-1], epoch*len(train_loader) + step)

                # Save Latest Model
                save_path = os.path.join(save_dir, args['dataset']+'_latest_model.pkl')
                save_checkpoint(epoch, step, model, optimizer, save_path)
                print('Lastest val accuracy checkpoint saved to: {}'.format(save_path))

            print('Accuracy of the network on the validation set: %f %%\n' % (100.*running_acc[-1]))

            # Save Best Validation Accuracy Model Separately
            if best_val_acc < running_acc[-1]:
                best_val_acc = running_acc[-1]

                if not args['debug']:
                    #Log best validation accuracy
                    if use_wandb:
                        wandb.run.summary['best_accuracy'] = best_val_acc

                    # Save Current Model
                    save_path = os.path.join(save_dir, args['dataset']+'_best_model.pkl')
                    save_checkpoint(epoch, step, model, optimizer, save_path)
                    print('Best val accuracy checkpoint saved to: {}'.format(save_path))

        if not args['debug']:
            # Close Tensorboard Element
            writer.close()

def valid(valid_loader, running_acc, model, model_loss, device):
    running_loss = 0.
    acc_metric = Metrics(**args, ndata=len(valid_loader.dataset), logger=wandb if use_wandb else None)
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            x_input     = data['data']
            annotations = data['annots'] 

            if isinstance(x_input, torch.Tensor):
                mini_batch_size = x_input.shape[0]
                outputs = model(x_input.to(device))
            else:
                mini_batch_size = x_input[0].shape[0] #Assuming the first element contains the true data input 
                for i, item in enumerate(x_input):
                    if isinstance(item, torch.Tensor):
                        x_input[i] = item.to(device)
                outputs = model(*x_input)
        
            running_acc.append(acc_metric.get_accuracy(outputs, annotations))

            if step % 100 == 0:
                print('Step: {}/{} | validation acc: {:.4f}'.format(step, len(valid_loader), running_acc[-1]))
    
    return running_acc


if __name__ == "__main__":

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])

    if not args['resume']:
        np.random.seed(args['seed'])

    train(**args)
