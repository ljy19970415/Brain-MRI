# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
#from sklearn.metrics import log_loss
import json
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel
from models import resnet,densenet

'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class MedKLIP(nn.Module):

    def __init__(self, config, ana_book, disease_book, mode='train'):
        super(MedKLIP, self).__init__()

        self.mode = mode
        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(ana_book['input_ids'].device)
            self.ana_book = bert_model(input_ids = ana_book['input_ids'],attention_mask = ana_book['attention_mask'])#(**encoded_inputs)
            self.ana_book = self.ana_book.last_hidden_state[:,0,:]
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]
        # self.disease_embedding_layer1 = nn.Linear(768,256)
        # self.disease_embedding_layer2 = nn.Linear(256,768)
        self.cl_fc = nn.Linear(config['out_feature'],768)
        self.excluded_disease = ['normal']
        self.disease_name = json.load(open('/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP/data_file/dis_order.json','r'))
        self.cl_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]   
        
        ''' visual backbone'''
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        # resnet = self._get_res_basemodel(config['res_base_model'])
        # num_ftrs = int(resnet.fc./2)
        # self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.res_l2 = nn.Linear(num_ftrs, self.d_model)
        if config['model_type']== 'resnet':
            resnet=self._get_resnet_model(config['model_type'],config['model_depth'],config['input_W'],
                                                config['input_H'],config['input_D'],config['resnet_shortcut'],
                                                config['no_cuda'],config['gpu_id'],config['pretrain_path'],config['out_feature'])
            num_ftrs=int(resnet.conv_seg[2].in_features)
            self.res_features = nn.Sequential(*list(resnet.children())[:-1])
            # num_ftrs=2048
            out_feature=config['out_feature']
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, out_feature)
            self.res_linear1=nn.Linear(out_feature*4,out_feature)
            self.res_linear2=nn.Linear(out_feature,out_feature)
        elif config['model_type'] == 'densenet':
            densenet=self._get_densenet_model(config)
            num_ftrs=int(densenet.classifier.in_features)
            self.res_features = nn.Sequential(*list(densenet.children())[:-1])
            # num_ftrs=2048
            out_feature=config['out_feature']
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, out_feature)
            self.res_linear1=nn.Linear(out_feature*4,out_feature)
            self.res_linear2=nn.Linear(out_feature,out_feature)


        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Parameter(torch.empty(config['num_classes'],self.d_model,config['attribute_set_size']))
        torch.nn.init.normal_(self.classifier, mean=0.0, std=0.02)

        self.apply(self._init_weights)

        # focal_loss_weight
        # self.class_weight = json.load(open(config['class_weight'],'r'))
        # LA
        disease_order=json.load(open(config['disease_order'],'r'))
        class_p = json.load(open(config['class_p'],'r'))
        self.class_p = torch.tensor(config["la_alpha"])*torch.log(torch.tensor([[class_p[i][0]/class_p[i][1]] for i in disease_order]))
        self.config = config

    def _get_resnet_model(self,model_type,model_depth,input_W,input_H,input_D,resnet_shortcut,no_cuda,gpu_id,pretrain_path,out_feature):
        assert model_type in [
            'resnet'
        ]

        if model_type == 'resnet':
            assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        if model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 256
        elif model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048

        model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=out_feature, bias=True))

        net_dict = model.state_dict()
        model = model.cuda()

        if pretrain_path != 'None':
            print('loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict) 
            model.load_state_dict(net_dict) 
            print("-------- pre-train model load successfully --------")
        return model
    
    def _get_densenet_model(self,config):
        assert config['model_type'] in [
            'densenet'
        ]

        if config['model_type'] == 'densenet':
            assert config['model_depth'] in [121,169,201,264]
            model=densenet.generate_model(model_depth=config['model_depth'],
                                        num_classes=config['out_feature'],
                                        n_input_channels=config['in_channels'],
                                        conv1_t_size=config['conv1_t_size'],
                                        conv1_t_stride=config['conv1_t_stride'],
                                        no_max_pool=config['no_max_pool'])
        return model


    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def image_encoder(self, image):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        
        img=image.float()
        img=img.cuda()
        batch_size = img.shape[0]
        res_fea = self.res_features(img) #batch_size,feature_size,patch_num,patch_num
        # print(res_fea.shape)
        res_fea = rearrange(res_fea,'b d n1 n2 n3 -> b (n1 n2 n3) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        
        return out_emb

    def forward(self, images,labels,smaple_index = None, is_train = True, no_cl= False, exclude_class= False):

        # labels batch,51,75 binary_label batch,75 sample_index batch,index
        B = images[0].shape[0]
        device = images[0].device
        ''' Visual Backbone '''
        # x = self.image_encoder(images) #batch_size,patch_num,dim

        # features = x.transpose(0,1) #patch_num b dim
        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        disease_book=self.disease_book.clone()
        ana_book=self.ana_book.clone()
        class_p = self.class_p.clone()
        disease_book=disease_book.to(device)
        ana_book=ana_book.to(device)
        class_p = class_p.to(device)
        # query_embed = self.disease_embedding_layer1(disease_book)
        # query_embed = self.disease_embedding_layer2(query_embed)
        query_embed = disease_book
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        features=[]
        ws_list=[]
        for i in range(4):
            feature=self.image_encoder(images[i])
            feature=feature.transpose(0,1)
            feature,ws = self.decoder(query_embed, feature, 
                memory_key_padding_mask=None, pos=None, query_pos=None)
            ws_mean=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4
            features.append(feature)
            ws_list.append(ws_mean)
        
        # no attention
        out_feature=torch.cat(features,dim=2) # 32,14,4*768
        out_feature=self.res_linear1(out_feature) 
        out_feature=self.res_linear2(out_feature)# 32,14,768
        out = self.dropout_feas(out_feature)

        if is_train == True and no_cl == False:
            anatomy_query = ana_book[smaple_index,:] # batch, 4 , dim
             # [Q,B,A]
            ll = out.transpose(0,1) # B Q A
            Q = ll.shape[1]
            ll = ll.reshape(ll.shape[0]*ll.shape[1],-1)
            ll = self.cl_fc(ll)
            ll = ll.unsqueeze(dim =-1)
            #ll = ll.reshape(B,Q,-1)
            anatomy_query = anatomy_query.reshape(B*Q,8,768)
            ll = torch.bmm(anatomy_query, ll ).squeeze()  # B Q 4
            cl_labels = torch.zeros((ll.shape[0])).to(device)
            #if exclude_class == True:
            cl_labels = cl_labels.reshape(B,Q)
            cl_labels = cl_labels[:,self.cl_class_dim]
            cl_labels = cl_labels.reshape(-1)
            ll = ll.reshape(B,Q,-1)
            ll = ll[:,self.cl_class_dim,:]
            ll = ll.reshape(B*(len(self.cl_class_dim)),-1)
        
        
        # class seperate classifier
        out = out.transpose(0,1) # b,14,768
        out = out.unsqueeze(-2) # b,14,1,768
        oB = out.shape[0] # b
        oC = out.shape[1] # 14
        out = rearrange(out,'b c l d -> (b c) l d') # b*14, 1, 768
        clas = self.classifier.repeat(oB,1,1,1) # b,14,768,1
        clas = rearrange(clas,'b c d l -> (b c) d l') # b*14, 768,1

        x = torch.bmm(out,clas) # b*14,1,1
        x = rearrange(x,'(b c) e f -> b c (e f)',b=oB,c=oC) # b, 14, 1
         
        if exclude_class == True:
            labels = labels[:,self.keep_class_dim]
            x = x[:,self.keep_class_dim,:]
        
        cl_mask_labels = labels[:,self.cl_class_dim]
        cl_mask_labels = cl_mask_labels.reshape(-1,1) # b*left_class_num,1
        
        B = labels.shape[0]
        labels = labels.reshape(-1,1) # b*class_num,1
        logits = x.reshape(-1, x.shape[-1])
        Mask = ((labels != -1) & (labels != 2)).squeeze()
        # print('label.shape',labels.shape)
        # print('logits.shape',logits.shape)

        cl_mask = (cl_mask_labels == 1).squeeze()

        if is_train == True:
            labels = labels[Mask].float()
            logits = logits[Mask]
            if self.config["la"]:
                class_p = class_p.unsqueeze(0).repeat(B,1,1)
                class_p = class_p.reshape(-1,class_p.shape[-1])
                logits = logits + class_p
            logits = torch.sigmoid(logits)
            # print("logits",logits)
            loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0]) # b*class_num,2 , b*class_num
            # loss_ce = rearrange(loss_ce,'(b c) -> b c',c=class_num)
            
            if no_cl == False:
                cl_labels = cl_labels[cl_mask].long()
                ll = ll[cl_mask]
                loss_cl = F.cross_entropy(ll,cl_labels)
                # print("loss_cl",loss_cl)
                loss = loss_ce +loss_cl  
            else:
                loss_cl = torch.tensor(0).to(device)
                loss = loss_ce
        else:
            loss = 0
        # print("loss",loss)
        if is_train==True:
            return loss,loss_ce,loss_cl,x
        else:
            return loss,x,ws


    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()