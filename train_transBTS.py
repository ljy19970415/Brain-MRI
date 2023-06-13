
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import utils

from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from dataset.BraTS import BraTS

from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.nn.functional as F

def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    for i in range(n_class):
        cur_gt = gt_np[:,i]
        cur_pred = pred_np[:,i]
        Mask = (( cur_gt!= -1) & ( cur_gt != 2)).squeeze()
        cur_gt = cur_gt[Mask]
        cur_pred = cur_pred[Mask]
        if (not 1 in cur_gt) or (not 0 in cur_gt):
            AUROCs.append(-1)
        else:
            AUROCs.append(roc_auc_score(cur_gt, cur_pred))
    return AUROCs

def evaluate(tensorboard):
    gt, pred = tensorboard["gt"],tensorboard["pred"]
    AUROCs = np.array(compute_AUCs(gt, pred,len(target_class)))
    max_f1s = []
    accs = []
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].detach().cpu().numpy() 
        Mask = (( gt_np!= -1) & ( gt_np != 2)).squeeze()
        gt_np = gt_np[Mask]
        pred_np = pred_np[Mask]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision # dot multiply for list
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
    return AUROCs,accs,max_f1s

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def train(model, data_loader, optimizer, epoch, device, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(loss_ce=1.0)
    metric_logger.update(lr = optimizer.param_groups[0]['lr'])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1   
    scalar_step = epoch*len(data_loader)

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    tensorboard = {
        "train_loss_ce":[],
        "gt":gt,
        "pred":pred
    }

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = sample['image'].to(device)
        labels = sample['label'].to(device)

        B = labels.shape[0]
        
        optimizer.zero_grad()

        logits = model(images)
        
        print(self.co)

        # logits.shape torch.Size([112, 1]) torch.Size([112, 1]) torch.Size([112])
        # ll.shape torch.Size([104, 8]) torch.Size([104]) torch.Size([104])

        tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

        B = labels.shape[0]

        # if "exclude_class" in config and config["exclude_class"]:
        #     labels = labels[:,keep_class_dim]
        
        pred_class = logits.reshape(-1,len(all_target_class))
        
        if config['num_classes']>13:
            label_former = labels[:,:13].reshape(-1,1)
            logit_former = pred_class[:,:13].reshape(-1,1)
            label_latter = labels[:,13:].reshape(-1,1)
            logit_latter = pred_class[:,13:].reshape(-1,1)
            # print("shape",label_former.shape,logit_former.shape,label_latter.shape,logit_latter.shape)
            Mask1 = ((label_former != -1) & (label_former != 2)).squeeze()
            label_former = label_former[Mask1].float()
            logit_former = logit_former[Mask1]
            Mask2 = ((label_latter != -1) & (label_latter != 2)).squeeze()
            label_latter = label_latter[Mask2].float()
            logit_latter = logit_latter[Mask2]
            loss_ce_former = F.binary_cross_entropy(logit_former[:,0],label_former[:,0])
            loss_ce_latter = F.binary_cross_entropy(logit_latter[:,0],label_latter[:,0])

        labels = labels.reshape(-1,1) # b*class_num,1

        Mask = ((labels != -1) & (labels != 2)).squeeze()

        labels = labels[Mask].float()
        logits = logits[Mask]

        loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0])
        
        tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
        tensorboard["train_loss_ce"].append(loss_ce.item())

        if config['num_classes']>13:
            tensorboard["train_loss_ce_former"].append(loss_ce_former.item())
            tensorboard["train_loss_ce_latter"].append(loss_ce_latter.item())

        loss_ce.backward()
        optimizer.step()
        torch.cuda.synchronize()
        
        scalar_step += 1
        metric_logger.update(loss_ce=loss_ce.item())     
        metric_logger.update(lr = optimizer.param_groups[0]['lr'])

    # get mean loss for the epoch
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean() if len(tensorboard[i]) else 0
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, tensorboard #,loss_epoch.mean()

def valid(model, data_loader, device, config):
    model.eval()

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    tensorboard = {
        "val_loss_ce":[],
        "gt":gt,
        "pred":pred
    }

    for i, sample in enumerate(data_loader):

        images = sample['image'].to(device)
        labels = sample['label'].to(device)

        B = labels.shape[0]
        
        with torch.no_grad():
            logits = model(images)
            print("logits.shape",logits.shape)
            print(self.cp)
            
            tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

            B = labels.shape[0]

            if "exclude_class" in config and config["exclude_class"]:
                labels = labels[:,keep_class_dim]

            pred_class = logits.reshape(-1,len(all_target_class))

            if config['num_classes']>13:
                label_former = labels[:,:13].reshape(-1,1)
                logit_former = pred_class[:,:13].reshape(-1,1)
                label_latter = labels[:,13:].reshape(-1,1)
                logit_latter = pred_class[:,13:].reshape(-1,1)
                Mask1 = ((label_former != -1) & (label_former != 2)).squeeze()
                label_former = label_former[Mask1].float()
                logit_former = logit_former[Mask1]
                Mask2 = ((label_latter != -1) & (label_latter != 2)).squeeze()
                label_latter = label_latter[Mask2].float()
                logit_latter = logit_latter[Mask2]
                loss_ce_former = F.binary_cross_entropy(logit_former[:,0],label_former[:,0])
                loss_ce_latter = F.binary_cross_entropy(logit_latter[:,0],label_latter[:,0])
            
            labels = labels.reshape(-1,1) # b*class_num,1

            Mask = ((labels != -1) & (labels != 2)).squeeze()

            labels = labels[Mask].float()
            logits = logits[Mask]

            loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0])
            
            # pred_class = pred_class[:,:,0]
            tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
            # val_loss.append(loss.item())
            tensorboard["val_loss_ce"].append(loss_ce.item())
            if config['num_classes']>13:
                tensorboard["val_loss_ce_former"].append(loss_ce_former.item())
                tensorboard["val_loss_ce_latter"].append(loss_ce_latter.item())
        
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean() if len(tensorboard[i]) else 0

    return tensorboard

def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['max_epochs']

    #### Dataset #### 
    print("Creating dataset")
    print("train file",config['train_file'])
    print("valid file",config['valid_file'])

    # batch_size = 16
    train_datasets = BraTS(config['train_file'],config['label_file'], mode = 'train')
    train_dataloader = DataLoader(
            train_datasets,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )
    
    val_datasets = BraTS(config['valid_file'],config['label_file'], mode ='train')
    val_dataloader = DataLoader(
            val_datasets,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )   

    print("Creating model")

    model = TransBTS(img_dim_x=config['input_W'], img_dim_y=config['input_H'], img_dim_z=config['input_D'], _conv_repr=True, _pe_type="learned")

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids) 
    model = model.cuda(device=device_ids[0])

    # arg_opt = utils.AttrDict(config['optimizer'])
    # optimizer = create_optimizer(arg_opt, model)
    # lr 0.0004 weight_decay 1e-5 amsgrad True
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
    # arg_sche = utils.AttrDict(config['schedular'])
    
    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))

    best_val_ce_loss = float("inf")

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        
        train_stats, tensorboard_train = train(model, train_dataloader, optimizer, epoch, device, config) 

        writer.add_scalar('lr/leaning_rate', optimizer.param_groups[0]['lr'], epoch)

        tensorboard_val = valid(model, val_dataloader, device, config)

        if config['num_classes']>13:
            content = {'train_loss':tensorboard_train["train_loss_ce"],\
                        "train_dis_loss":tensorboard_train["train_loss_ce_former"],\
                        "train_other_loss":tensorboard_train["train_loss_ce_latter"],\
                        "val_loss":tensorboard_val["val_loss_ce"],\
                        "val_dis_loss":tensorboard_val["val_loss_ce_former"],\
                        "val_other_loss":tensorboard_val["val_loss_ce_latter"]}
            writer.add_scalars('loss_ce_epoch',content, epoch)
        else:
            writer.add_scalars('loss_ce_epoch',{'train_loss':tensorboard_train["train_loss_ce"],"val_loss":tensorboard_val["val_loss_ce"]}, epoch)

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': tensorboard_val["val_loss"]
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        train_metric = evaluate(tensorboard_train)
        val_metric = evaluate(tensorboard_val)
        train_roc = {i:train_metric[0][idx] for idx,i in enumerate(target_class)}
        train_acu = {i:train_metric[1][idx] for idx,i in enumerate(target_class)}
        train_f1 = {i:train_metric[2][idx] for idx,i in enumerate(target_class)}
        val_roc = {i:val_metric[0][idx] for idx,i in enumerate(target_class)}
        val_acu = {i:val_metric[1][idx] for idx,i in enumerate(target_class)}
        val_f1 = {i:val_metric[2][idx] for idx,i in enumerate(target_class)}
        writer.add_scalars('train_metric/roc',train_roc, epoch)
        writer.add_scalars('train_metric/acu',train_acu, epoch)
        writer.add_scalars('train_metric/f1',train_f1, epoch)
        writer.add_scalars('val_metric/roc',val_roc, epoch)
        writer.add_scalars('val_metric/acu',val_acu, epoch)
        writer.add_scalars('val_metric/f1',val_f1, epoch)

        # if epoch % 10 == 1 and epoch>1:
        if utils.is_main_process() and best_val_ce_loss > tensorboard_val["val_loss_ce"]:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'best_val.pth')) 
            best_val_ce_loss = tensorboard_val["val_loss_ce"]
            print("save best",epoch)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP_bce/configs_baseline/transBTS.yaml')
    parser.add_argument('--finetune_checkpoint', default='')
    parser.add_argument('--output_dir', default='outputdir_transBTS')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='4,5,6,7', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True
    all_target_class = json.load(open(config['disease_order'],'r'))
    target_class = all_target_class[:13].copy()
    # if "exclude_class" in config and config["exclude_class"]:
    #     keep_class_dim = [all_target_class.index(i) for i in all_target_class if i not in config["exclude_classes"] ]
    #     all_target_class = [target_class[i] for i in keep_class_dim]
    #     keep_class_dim = [target_class.index(i) for i in target_class if i not in config["exclude_classes"] ]
    #     target_class = [target_class[i] for i in keep_class_dim]
    main(args, config)