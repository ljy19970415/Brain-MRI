import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import json
import nibabel as nib
from dataset.augment import *

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        # label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            # label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            # label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            # label = np.flip(label, 2)
        sample['image'] = image
        return sample


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        H = random.randint(0, 224 - 128)
        W = random.randint(0, 224 - 128)

        image = image[H: H + 128, W: W + 128, ..., ...]
        sample['image'] = image
        return sample


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        # label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor
        sample['image'] = image

        return sample


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        # label = sample['label']
        # label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        # label = torch.from_numpy(label).long()
        sample['image'] = image
        return sample


def transform(sample):
    trans = transforms.Compose([
        # Pad(),
        # Random_rotate(),  # time-consuming
        # Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)

def nib_load(file_name, component):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    if data.ndim>3:
        data=data[:,:,:,component]
    proxy.uncache()
    return data

class BraTS(Dataset):
    def __init__(self, csv_path, np_path, mode = 'train'):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)
        self.mode=mode
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        images = []
        for modal in modal_dic:
            data = np.array(nib_load(self.ann[fid][modal], self.ann[fid]['component']), dtype='float32', order='C')
            # image=self.transform(img_data)
            if self.mode == 'train':
                image = nnUNet_resample(data,[224,224,24],is_seg=False)
                images.append(image)
        images = np.stack(images, -1)
    
        mask = images.sum(-1) > 0
        for k in range(4):
            x = images[..., k]  #
            y = x[mask]

            x[mask] -= y.mean()
            x[mask] /= y.std()

            images[..., k] = x
        
        class_label = self.rad_graph_results[self.ann[fid]["labels_id"],:,:] # (51, 75)
        labels = self.triplet_extraction(class_label)

        sample = {'image': images, 'label': labels, "fid":fid}
        if self.mode == "train":
            sample = transform(sample)
        elif self.mode == "val":
            sample = transform_valid(sample)
        return sample
    
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1
        for i in range(class_label.shape[1]):
            ### extract the exist label for each entity and maintain -1 if not mentioned. ###
            if 0 in class_label[:,i]:
                exist_labels[i] = 0
                
            if 1 in class_label[:,i]:
                exist_labels[i] = 1

        return exist_labels
    
    def __len__(self):
        return len(self.fid_list)



