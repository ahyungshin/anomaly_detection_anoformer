import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn

def load_data_power(opt):

    #- data
    with open(str(os.path.join(opt.dataroot, "power_data_train.pkl")), 'rb') as f:
        N_samples = torch.FloatTensor(pickle.load(f)) 
        N_samples_y = N_samples[:,-1] # 18145
        N_samples = N_samples[:,:-1] # 18145,1

    with open(str(os.path.join(opt.dataroot, "power_data_test.pkl")), 'rb') as f:
        data = torch.FloatTensor(pickle.load(f)) 
        test_label = data[:,-1] # 14768
        test_data = data[:,:-1] # 14768,1

    N_samples = normalize(N_samples) 
    test_data = normalize(test_data) 

    N_samples = N_samples.unfold(0,100, 1) # [18046, 1, 100]
    N_samples_y = N_samples_y.unfold(0,100, 1) # [18046, 1, 100]
    test_data = test_data.unfold(0,100, 1) # [18046, 1, 100]
    test_label = test_label.unfold(0,100, 1) # [18046, 1, 100]

    # test val split
    test_AN, val_AN, test_AN_y, val_AN_y = getPercent(test_data, test_label, 0.3, 0)
    N_samples_y = torch.any(N_samples_y, axis=-1).type(torch.float32)
    val_AN_y = torch.any(val_AN_y, axis=-1).type(torch.float32)
    test_AN_y = torch.any(test_AN_y, axis=-1).type(torch.float32)


    print("train data size:{}".format(N_samples.shape))
    print("val data size:{}".format(val_AN.shape))
    print("test data size:{}".format(test_AN.shape))

    train_dataset = TensorDataset(torch.Tensor(N_samples),torch.Tensor(N_samples_y))
    val_dataset= TensorDataset(torch.Tensor(val_AN), torch.Tensor(val_AN_y))
    test_dataset = TensorDataset(torch.Tensor(test_AN), torch.Tensor(test_AN_y))
        

    dataloader = {"train":DataLoader(
                            dataset=train_dataset,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            num_workers=int(opt.workers),
                            drop_last=True),
                "val":DataLoader(
                            dataset=val_dataset, 
                            batch_size=opt.batchsize,
                            shuffle=True,
                            num_workers=int(opt.workers),
                            drop_last=False),
                "test":DataLoader(
                            dataset=test_dataset,
                            batch_size=opt.batchsize, 
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False),
                    }
    return dataloader

    
def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-torch.min(seq))/(torch.max(seq)-torch.min(seq))-1


def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y
