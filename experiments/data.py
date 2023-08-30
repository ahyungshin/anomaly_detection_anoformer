import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def load_data(opt):

    #- data
    N_samples =np.load(os.path.join(opt.dataroot, "normal.npy")) #NxCxL
    AN_samples = np.load(os.path.join(opt.dataroot, "ano1.npy")) #NxCxL

    #- label
    N_samples_y = np.load(os.path.join(opt.dataroot, "normal_label.npy")) #NxCxL
    AN_samples_y = np.load(os.path.join(opt.dataroot, "ano1_label.npy")) #NxCxL
    
    # normalize all
    for i in range(N_samples.shape[0]):
        for j in range(opt.nc):
            N_samples[i][j]=normalize(N_samples[i][j][:])
    N_samples=N_samples[:,:opt.nc,:]

    for i in range(AN_samples.shape[0]):
        for j in range(opt.nc):
            AN_samples[i][j] = normalize(AN_samples[i][j][:])
    AN_samples = AN_samples[:, :opt.nc, :]

    # test val split
    test_AN, val_AN, test_AN_y, val_AN_y = getPercent(AN_samples, AN_samples_y, 0.3, 0)
    N_samples_y = np.any(N_samples_y, axis=-1)
    val_AN_y = np.any(val_AN_y, axis=-1)
    test_AN_y = np.any(test_AN_y, axis=-1)


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
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1


def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y
