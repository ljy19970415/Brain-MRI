
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
from optim.optim_factory_kad import create_optimizer
from dataset.dataset import MedKLIP_Dataset
# from models.model_MedKLIP import MedKLIP
from models.model_MedKLIP_before_fuse import MedKLIP as MedKLIP
from models.model_MedKLIP_14class_before_fuse import MedKLIP as MedKLIP_14
from models.before_fuse import *
# from models.model_MedKLIP_attention_14class import MedKLIP as MedKLIP_14_atten

from models.tokenization_bert import BertTokenizer

from models.imageEncoder import ModelRes, ModelDense
from transformers import AutoModel

from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.nn.functional as F
from loss.loss import *
from dataset.sampler import *


def get_tokenizer(tokenizer,target_text):
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
    return target_tokenizer

def get_text_features(model,text_list,tokenizer,device,max_length):
    # text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    target_tokenizer = tokenizer(list(text_list), padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(device)
    # text_features = model.encode_text(text_token)
    text_features = model(input_ids = target_tokenizer['input_ids'],attention_mask = target_tokenizer['attention_mask'])#(**encoded_inputs)
    text_features = text_features.last_hidden_state[:,0,:]
    # text_features = F.normalize(text_features, dim=-1)
    return text_features

def gen_entity_labels(entity):
    # entity [(b,1),(b,1),...]
    entity_labels = []
    for group in entity:
        b = len(group)
        a=np.eye(b)
        for i in range(b):
            text1 = sorted(group[i].split('[SEP]'))
            if len(text1) == 1 and text1[0] == "unspecified":
                continue
            for j in range(i+1,b):
                text2 = sorted(group[j].split('[SEP]'))
                a[i][j] = 1 if len(text1) == len(text2) and text1 == text2 else 0
                a[j][i] = a[i][j]
        c=1/a.sum(axis=-1)
        d = np.array([a[idx]*c[idx] for idx in range(len(a))])
        entity_labels.append(d)
    return torch.tensor(np.array(entity_labels))

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model


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

def train(model, image_encoder, text_encoder, fuseModule, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    clip_loss = ClipLoss()
    # asl_loss = ASLLoss(config) if "asl_loss" in config and config["asl_loss"] else None
    global_local_loss = GlobalLocalLoss() if 'global_local_loss' in config and config['global_local_loss'] else None
    model.train()
    if config['4_image_encoder']:
        for idx in range(len(image_encoder)):
            image_encoder[idx].train()
    else:
        image_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(loss_ce=1.0)
    metric_logger.update(loss_cl=1.0)
    metric_logger.update(loss_clip=1.0)
    metric_logger.update(loss_global=1.0)
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
        "train_loss_ce_former":[],
        "train_loss_ce_latter":[],
        "train_loss_cl":[],
        "train_loss_clip":[],
        "train_loss_global_local":[],
        "gt":gt,
        "pred":pred
    }

    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    ana_order=json.load(open(config['anatomy_order'],'r'))
    ana_book = [ 'It is located at ' + i for i in ana_order]

    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=128)
    ana_features = get_text_features(text_encoder,ana_book,tokenizer,device,max_length=128)

    cl_excluded_disease = ['normal']
    if "exclude_class" in config and config["exclude_class"]:
        cl_excluded_disease += config["exclude_classes"]
        keep_class_dim = [json_order.index(i) for i in json_order if i not in config["exclude_classes"] ]   
    cl_class_dim = [json_order.index(i) for i in json_order if i not in cl_excluded_disease]


    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = sample['image']  # [(b,x,y,z),(b,x,y,z)]
        labels = sample['label'].to(device)
        index = sample['index'].to(device)
        entity = sample['entity'] # [(b,a1),(b, a2),...]
        report = sample["report"] # [b,whole]

        B = labels.shape[0]

        cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)
        cur_ana_features = ana_features.unsqueeze(0).repeat(B,1,1)
        entity_labels = gen_entity_labels(entity).float().to(device)
        entity_features = []
        for en in entity:
            entity_features.append(get_text_features(text_encoder,en,tokenizer,device,max_length=128)) # [(b,d),(b,d)]
        
        if 'global_local_loss' in config and config['global_local_loss']:
            report_feature = get_text_features(text_encoder,report,tokenizer,device,max_length=128) # b d
            report_labels = torch.tensor(np.eye(B)).float().to(device)
        
        optimizer.zero_grad()
        
        # entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)
        image_features = [] # image_features 4 b n d, image_features_pool 4 b d
        image_features_pool = []
        for idx,cur_image in enumerate(images):
            if config['4_image_encoder']:
                cur_image_encoder = image_encoder[idx]
                image_feature,image_feature_pool = cur_image_encoder(cur_image)
            else:
                image_feature,image_feature_pool = image_encoder(cur_image) 
            image_features.append(image_feature)
            image_features_pool.append(image_feature_pool)
        
        # before fuse
        fuse_image_feature, fuse_image_feature_pool = fuseModule(image_features)
        image_features_pool.append(fuse_image_feature_pool)


        if config['no_cl'] == False:
            logits, ll, cl_labels = model(fuse_image_feature,cur_text_features,cur_ana_features,index)
        else:
            logits = model(fuse_image_feature,cur_text_features,cur_ana_features,index)
        # logits.shape torch.Size([112, 1]) torch.Size([112, 1]) torch.Size([112])
        # ll.shape torch.Size([104, 8]) torch.Size([104]) torch.Size([104])

        tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

        cl_mask_labels = labels[:,cl_class_dim]
        cl_mask_labels = cl_mask_labels.reshape(-1,1) # b*left_class_num,1

        B = labels.shape[0]

        if "exclude_class" in config and config["exclude_class"]:
            labels = labels[:,keep_class_dim]
        
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

        # cur_target_class= target_class[Mask]

        # if config["la"]:
        #     class_p = class_p.unsqueeze(0).repeat(B,1,1)
        #     class_p = class_p.reshape(-1,class_p.shape[-1])
        #     logits = logits + class_p

        loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0])

        cl_mask = (cl_mask_labels == 1).squeeze()
        if config['no_cl'] == False:
            cl_labels = cl_labels[cl_mask].long()
            ll = ll[cl_mask]
            loss_cl = F.cross_entropy(ll,cl_labels)
        else:
            loss_cl = torch.tensor(0).to(device)

        # if args.use_entity_features:
        #     pred_class_text = model(entity_features.unsqueeze(1),text_features)
        #     loss_ce_text = ce_loss(pred_class_text.view(-1,2),label.view(-1))
        #     loss_ce = loss_ce_image + loss_ce_text
        # else:
        #     loss_ce = loss_ce_image

        if config['kad']:
            loss_clip = clip_loss(image_features_pool, entity_features, entity_labels) # 4 b d; 4 b d; 4 b b
        else:
            loss_clip = torch.tensor(0).to(device)
        
        if 'global_local_loss' in config and config['global_local_loss']:
            loss_global_local = global_local_loss(image_features_pool, report_feature, report_labels)
            global_local_loss_ratio = config['global_local_loss_ratio']
        else:
            loss_global_local = torch.tensor(0).to(device)
            global_local_loss_ratio = torch.tensor(0).to(device)

        
        loss = loss_ce + loss_cl + loss_clip * config['kad_loss_ratio'] + loss_global_local * global_local_loss_ratio

        
        # pred_class = logits.reshape(-1,len(target_class))
        # pred_class = pred_class[:,:,0]
        tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
        tensorboard["train_loss"].append(loss.item())
        tensorboard["train_loss_ce"].append(loss_ce.item())
        if config['no_cl'] == False:
            tensorboard["train_loss_cl"].append(loss_cl.item())
        if config['kad']:
            tensorboard["train_loss_clip"].append(loss_clip.item())
        if 'global_local_loss' in config and config['global_local_loss']:
            tensorboard["train_loss_global_local"].append(loss_global_local.item())
        if config['num_classes']>13:
            tensorboard["train_loss_ce_former"].append(loss_ce_former.item())
            tensorboard["train_loss_ce_latter"].append(loss_ce_latter.item())

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        
        scalar_step += 1
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_clip=loss_clip.item())
        metric_logger.update(loss_global=loss_global_local.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])


    # get mean loss for the epoch
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean() if len(tensorboard[i]) else 0
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, tensorboard #,loss_epoch.mean()

def valid(model, image_encoder, text_encoder, fuseModule, tokenizer, data_loader, device,config):
    clip_loss = ClipLoss()
    global_local_loss = GlobalLocalLoss() if 'global_local_loss' in config and config['global_local_loss'] else None
    model.eval()
    if config['4_image_encoder']:
        for idx in range(len(image_encoder)):
            image_encoder[idx].eval()
    else:
        image_encoder.eval()
    # temp = nn.Parameter(torch.ones([]) * config['temp'])   
    # val_scalar_step = epoch*len(data_loader)
    # val_loss = []
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    tensorboard = {
        "val_loss":[],
        "val_loss_ce":[],
        "val_loss_ce_former":[],
        "val_loss_ce_latter":[],
        "val_loss_cl":[],
        "val_loss_clip":[],
        "val_loss_global_local":[],
        "gt":gt,
        "pred":pred
    }

    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    ana_order=json.load(open(config['anatomy_order'],'r'))
    ana_book = [ 'It is located at ' + i for i in ana_order]

    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=128)
    ana_features = get_text_features(text_encoder,ana_book,tokenizer,device,max_length=128)

    cl_excluded_disease = ['normal']
    if "exclude_class" in config and config["exclude_class"]:
        cl_excluded_disease += config["exclude_classes"]
        keep_class_dim = [json_order.index(i) for i in json_order if i not in config["exclude_classes"] ]   
    cl_class_dim = [json_order.index(i) for i in json_order if i not in cl_excluded_disease]  

    for i, sample in enumerate(data_loader):

        images = sample['image']  # [(b,x,y,z),(b,x,y,z)]
        labels = sample['label'].to(device)
        index = sample['index'].to(device)
        entity = sample['entity'] # [(b,a1),(b, a2),...]
        report = sample["report"] # [b,whole]

        B = labels.shape[0]
        
        with torch.no_grad():
            cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)
            cur_ana_features = ana_features.unsqueeze(0).repeat(B,1,1)
            entity_labels = gen_entity_labels(entity).float().to(device)
            entity_features = []
            for en in entity:
                entity_features.append(get_text_features(text_encoder,en,tokenizer,device,max_length=128)) # [(b,d),(b,d)]
            
            if 'global_local_loss' in config and config['global_local_loss']:
                report_feature = get_text_features(text_encoder,report,tokenizer,device,max_length=128) # b d
                report_labels = torch.tensor(np.eye(B)).float().to(device)

            image_features = [] # image_features 4 b n d, image_features_pool 4 b d
            image_features_pool = []
            for idx,cur_image in enumerate(images):
                if config['4_image_encoder']:
                    cur_image_encoder = image_encoder[idx]
                    image_feature,image_feature_pool = cur_image_encoder(cur_image)
                else:
                    image_feature,image_feature_pool = image_encoder(cur_image) 
                image_features.append(image_feature)
                image_features_pool.append(image_feature_pool)
            
            # before fuse
            fuse_image_feature, fuse_image_feature_pool = fuseModule(image_features)
            image_features_pool.append(fuse_image_feature_pool)
            
            if config['no_cl'] == False:
                logits, ll, cl_labels = model(fuse_image_feature,cur_text_features,cur_ana_features,index)
            else:
                logits = model(fuse_image_feature,cur_text_features,cur_ana_features,index)
            
            tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

            cl_mask_labels = labels[:,cl_class_dim]
            cl_mask_labels = cl_mask_labels.reshape(-1,1) # b*left_class_num,1

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

            cl_mask = (cl_mask_labels == 1).squeeze()

            if config['no_cl'] == False:
                cl_labels = cl_labels[cl_mask].long()
                ll = ll[cl_mask]
                loss_cl = F.cross_entropy(ll,cl_labels)
            else:
                loss_cl = torch.tensor(0).to(device)

            if config['kad']:
                loss_clip = clip_loss(image_features_pool, entity_features, entity_labels) # 4 b d; 4 b d; 4 b b
            else:
                loss_clip = torch.tensor(0).to(device)
            
            if 'global_local_loss' in config and config['global_local_loss']:
                loss_global_local = global_local_loss(image_features_pool, report_feature, report_labels)
                global_local_loss_ratio = config['global_local_loss_ratio']
            else:
                loss_global_local = torch.tensor(0).to(device)
                global_local_loss_ratio = torch.tensor(0).to(device)
    
            loss = loss_ce + loss_cl + loss_clip * config['kad_loss_ratio'] + loss_global_local * global_local_loss_ratio

            
            # pred_class = pred_class[:,:,0]
            tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
            # val_loss.append(loss.item())
            tensorboard["val_loss"].append(loss.item())
            tensorboard["val_loss_ce"].append(loss_ce.item())
            if config['no_cl'] == False:
                tensorboard["val_loss_cl"].append(loss_cl.item())
            if config['kad']:
                tensorboard["val_loss_clip"].append(loss_clip.item())
            if 'global_local_loss' in config and config['global_local_loss']:
                tensorboard["val_loss_global_local"].append(loss_global_local.item())
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
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    print("train file",config['train_file'])
    print("valid file",config['valid_file'])
    train_datasets = MedKLIP_Dataset(config['train_file'],config['label_file'],config['report_observe'], mode = 'train')
    train_sampler = UniformSampler(train_datasets,config['batch_size'],batch_clas_num=8) if 'uniform_sample' in config and config['uniform_sample'] else None
    shuffle = False if 'uniform_sample' in config and config['uniform_sample'] else True
    train_dataloader = DataLoader(
            train_datasets,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=shuffle,
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

    print("Creating model")

    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    if config['model_type']== 'resnet':
        image_encoder = ModelRes(config).to(device)
    elif config['model_type'] == 'densenet':
        image_encoder = ModelDense(config).to(device)

    text_encoder = _get_bert_basemodel(config['text_encoder']).to(device)

    if config['seperate_classifier']:
        print("Medklip_14")
        model = MedKLIP_14(config)
    else:
        print("medklip")
        model = MedKLIP(config)
    
    fuseModule = beforeFuse(config).to(device) # before fusion

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids) 
    model = model.cuda(device=device_ids[0])

    if config['model_depth'] == 50:
        image_encoder = nn.DataParallel(image_encoder, device_ids) 
        image_encoder = image_encoder.cuda(device=device_ids[0])

    if len(args.finetune_checkpoint):    
        checkpoint = torch.load(args.finetune_checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            if "classifier" in name:
                print(name)
                param.requires_grad = True
                print("init",name)
                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.constant_(param,0)
                else:
                    print("param.shape",param.shape)
                    for i in range(len(param)):
                        torch.nn.init.normal_(param[i], mean=0.0, std=0.02)
            else:
                param.requires_grad = False 
        print('load finetune checkpoint from %s'%args.finetune_checkpoint)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, image_encoder, text_encoder)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))

    best_val_ce_loss = float("inf")

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        
        train_stats, tensorboard_train = train(model, image_encoder, text_encoder, fuseModule, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, config) 

        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        tensorboard_val = valid(model, image_encoder, text_encoder, fuseModule, tokenizer, val_dataloader, device, config)

        writer.add_scalars('loss_epoch',{'train_loss':tensorboard_train["train_loss"],"val_loss":tensorboard_val["val_loss"]}, epoch)
        
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
        
        if config['no_cl'] == False:
            writer.add_scalars('loss_cl_epoch',{'train_loss':tensorboard_train["train_loss_cl"],"val_loss":tensorboard_val["val_loss_cl"]}, epoch)
        if config['kad']:
            writer.add_scalars('loss_clip_epoch',{'train_loss':tensorboard_train["train_loss_clip"],"val_loss":tensorboard_val["val_loss_clip"]}, epoch)
        if 'global_local_loss' in config and config['global_local_loss']:
            writer.add_scalars('loss_global_local_epoch',{'train_loss':tensorboard_train["train_loss_global_local"],"val_loss":tensorboard_val["val_loss_global_local"]}, epoch)

        if utils.is_main_process():  
            image_encoder_params = [cur.state_dict() for cur in image_encoder] if config['4_image_encoder'] else image_encoder.state_dict()
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': tensorboard_val["val_loss"]
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder_params,
                'fuseModule': fuseModule.state_dict(),
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

        # if epoch % 10 == 1 and epoch>1:
        if utils.is_main_process() and best_val_ce_loss > tensorboard_val["val_loss_ce"]:
            image_encoder_params = image_encoder.state_dict()
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder_params,
                'fuseModule': fuseModule.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
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
    parser.add_argument('--config', default='/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP_bce/configs/Pretrain_MedKLIP_glrep_lesscls_fusereport_nocl.yaml')
    parser.add_argument('--finetune_checkpoint', default='')
    parser.add_argument('--output_dir', default='outputdir_34_kad_sc_glrep_lesscls_beforefuse_modalwise_fusewise_nocl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='1,2,3,0', help='gpu')
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
    if "exclude_class" in config and config["exclude_class"]:
        keep_class_dim = [all_target_class.index(i) for i in all_target_class if i not in config["exclude_classes"] ]
        all_target_class = [target_class[i] for i in keep_class_dim]
        keep_class_dim = [target_class.index(i) for i in target_class if i not in config["exclude_classes"] ]
        target_class = [target_class[i] for i in keep_class_dim]
    main(args, config)