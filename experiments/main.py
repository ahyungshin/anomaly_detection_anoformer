import torch

from options import Options
from data import load_data
from data_mit import load_data_mit
from data_power import load_data_power
from data_gesture import load_data_gesture
from utils import seed_all
seed_all()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = Options().parse()
print(opt)

if opt.dataset == 'mit_bih':
    dataloader=load_data_mit(opt)
    print("load Mit-BIH data success!!!")
elif opt.dataset == 'neurips_ts':
    dataloader=load_data(opt)
    print("load NeurIPS-TS data success!!!")
elif opt.dataset == 'power_data':
    dataloader=load_data_power(opt)
    print("load Power-Demand data success!!!")
elif opt.dataset == 'ann_gun_CentroidA':
    dataloader=load_data_gesture(opt)
    print("load 2D-Gesture data success!!!")
else:
    raise Exception("no this dataset :{}".format(opt.dataset))


if opt.model == "anoformer":
    from model import AnoFormer as MyModel
else:
    raise Exception("no this model :{}".format(opt.model))

model=MyModel(opt,dataloader,device)


if not opt.istest:
    print("################  Train  ##################")
    model.train()
else:
    print("################  Eval  ##################")
    model.load()
    model.test_type_neurips_ts()