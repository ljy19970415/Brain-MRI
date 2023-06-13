
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
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.dataset import MedKLIP_Dataset
# from models.model_MedKLIP import MedKLIP
from models.model_MedKLIP import MedKLIP as MedKLIP 
from models.model_MedKLIP_14class import MedKLIP as MedKLIP_14
from models.model_MedKLIP_attention_14class import MedKLIP as MedKLIP_14_atten

from models.tokenization_bert import BertTokenizer

from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.nn.functional as F

target_class=json.load(open('/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP/data_file/dis_order.json','r'))

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
    
    return target_tokenizer


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    for i in range(n_class):
        if (not 1 in gt_np[:,i]) or (not 0 in gt_np[:i]):
            AUROCs.append(-1)
        else:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def evaluate(tensorboard):
    gt, pred = tensorboard["gt"],tensorboard["pred"]
    AUROCs = np.array(compute_AUCs(gt, pred,len(target_class)))
    max_f1s = []
    accs = []
    for i in range(len(target_class)):   
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].detach().cpu().numpy()  
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision # dot multiply for list
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
    return AUROCs,accs,max_f1s

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(loss_ce=1.0)
    metric_logger.update(loss_cl=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    tensorboard = {
        "train_loss":[],
        "train_loss_ce":[],
        "train_loss_cl":[],
        "gt":gt,
        "pred":pred
    }

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        images = sample['image']
        labels = sample['label'].to(device)
        index = sample['index'].to(device)
        
        #print("image",len(images),"labels",labels.shape,"index",index.shape)
        optimizer.zero_grad()
        loss,loss_ce,loss_cl,x = model(images,labels, index, is_train= True,no_cl = config['no_cl'],exclude_class = config['exclude_class'])
        # loss has four value for each gpu
        loss=loss.sum()
        loss_ce=loss_ce.sum()
        loss_cl=loss_cl.sum()
        tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)
        pred_class = torch.sigmoid(x.reshape(-1,1)).reshape(-1,len(target_class))
        # pred_class = pred_class[:,:,0]
        tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
        tensorboard["train_loss"].append(loss.item()/len(labels))
        tensorboard["train_loss_ce"].append(loss_ce.item()/len(labels))
        tensorboard["train_loss_cl"].append(loss_cl.item()/len(labels))
        loss.backward()
        optimizer.step()   
        torch.cuda.synchronize()
        # writer.add_scalar('loss/loss', loss, scalar_step)
        # writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        # writer.add_scalar('loss/loss_cl', loss_cl, scalar_step)
        # for _, loss in enumerate(loss):    
        #     writer.add_scalar('loss/loss', loss.mean(), scalar_step)
        # for _, loss_ce in enumerate(loss_ce):  
        #     writer.add_scalar('loss/loss_ce', loss_ce.mean(), scalar_step)
        # for _, loss_cl in enumerate(loss_cl):  
        #     writer.add_scalar('loss/loss_cl', loss_cl.mean(), scalar_step)
        scalar_step += 1
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_cl=loss_cl.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    # get mean loss for the epoch
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, tensorboard #,loss_epoch.mean()


def valid(model, data_loader, epoch, device,config,writer):
    model.eval()
    temp = nn.Parameter(torch.ones([]) * config['temp'])   
    val_scalar_step = epoch*len(data_loader)
    #val_loss = []
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    tensorboard = {
        "val_loss":[],
        "val_loss_ce":[],
        "val_loss_cl":[],
        "gt":gt,
        "pred":pred
    }
    for i, sample in enumerate(data_loader):
        
        images = sample['image']
        labels = sample['label'].to(device)
        index = sample['index'].to(device)
        
        with torch.no_grad():
            loss,loss_ce,loss_cl,x = model(images,labels, index, is_train= True,no_cl = config['no_cl'],exclude_class = config['exclude_class'])
            loss=loss.sum()
            loss_ce=loss_ce.sum()
            loss_cl=loss_cl.sum()
            tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)
            pred_class = torch.sigmoid(x.reshape(-1,1)).reshape(-1,len(target_class))
            # pred_class = pred_class[:,:,0]
            tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
            #val_loss.append(loss.item())
            tensorboard["val_loss"].append(loss.item()/len(labels))
            tensorboard["val_loss_ce"].append(loss_ce.item()/len(labels))
            tensorboard["val_loss_cl"].append(loss_cl.item()/len(labels))
            # writer.add_scalar('val_loss/loss', loss, val_scalar_step)
            # writer.add_scalar('val_loss/loss_ce', loss_ce, val_scalar_step)
            # writer.add_scalar('val_loss/loss_cl', loss_cl, val_scalar_step)
            # for _, loss in enumerate(loss): 
            #     writer.add_scalar('val_loss/loss', loss.mean(), val_scalar_step)
            # for _, loss_ce in enumerate(loss_ce): 
            #     writer.add_scalar('val_loss/loss_ce', loss_ce.mean(), val_scalar_step)
            # for _, loss_cl in enumerate(loss_cl): 
            #     writer.add_scalar('val_loss/loss_cl', loss_cl.mean(), val_scalar_step)
            val_scalar_step += 1
    #avg_val_loss = np.array(val_loss).mean()
    # get mean loss for the epoch
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean()
    #return avg_val_loss
    return tensorboard

def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    print("train file",config['train_file'])
    print("valid file",config['valid_file'])
    train_datasets = MedKLIP_Dataset(config['train_file'],config['label_file'],config['report_observe'], mode = 'train')
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
    
    val_datasets = MedKLIP_Dataset(config['valid_file'],config['label_file'],config['report_observe'],mode ='train')
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

    print("Creating book")
    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    ana_order=json.load(open(config['anatomy_order'],'r'))
    ana_book = [ 'It is located at ' + i for i in ana_order]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    ana_book_tokenizer = get_tokenizer(tokenizer,ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    print("Creating model")
    if config['seperate_classifier'] and config['attention']:
        print("Medklip_14_attention")
        model = MedKLIP_14_atten(config,ana_book_tokenizer, disease_book_tokenizer, mode = 'train')
    elif config['seperate_classifier']:
        print("Medklip_14")
        model = MedKLIP_14(config,ana_book_tokenizer, disease_book_tokenizer, mode = 'train')
    else:
        print("medklip")
        model = MedKLIP(config,ana_book_tokenizer, disease_book_tokenizer, mode = 'train')
    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids) 
    model = model.cuda(device=device_ids[0])

    if len(args.finetune_checkpoint):    
        checkpoint = torch.load(args.finetune_checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                print("init",name)
                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.constant_(param,0)
            else:
                param.requires_grad = False 
        print('load finetune checkpoint from %s'%args.finetune_checkpoint)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats, tensorboard_train = train(model, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        # for k, v in train_stats.items():
        #     train_loss_epoch = v
        
        # writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        # writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        # writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_epoch), epoch)
        # writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)

        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        tensorboard_val = valid(model, val_dataloader, epoch,device,config,writer)
        #writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
        writer.add_scalars('loss_epoch',{'train_loss':tensorboard_train["train_loss"],"val_loss":tensorboard_val["val_loss"]}, epoch)
        writer.add_scalars('loss_ce_epoch',{'train_loss':tensorboard_train["train_loss_ce"],"val_loss":tensorboard_val["val_loss_ce"]}, epoch)
        writer.add_scalars('loss_cl_epoch',{'train_loss':tensorboard_train["train_loss_cl"],"val_loss":tensorboard_val["val_loss_cl"]}, epoch)
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': tensorboard_val["val_loss"]
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
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


        if epoch % 10 == 1 and epoch>1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))  

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP_bce/configs/Pretrain_MedKLIP_bce.yaml')
    parser.add_argument('--finetune_checkpoint', default='') 
    parser.add_argument('--output_dir', default='/home/ps/leijiayu/CODE/MedKLIP/outputdir_test')
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

    main(args, config)