import time,os
import random
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.optim as optim

from network import AD_MODEL,print_network
from transformer import Generator_TransformerModel, Discriminator_TransformerModel
from metric import evaluate


class AnoFormer(AD_MODEL):
    def __init__(self, opt, dataloader, device):
        super(AnoFormer, self).__init__(opt, dataloader, device)
        
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.niter = opt.niter
        
        self.mask_rate = self.opt.mask_rate
        self.mask_len = self.opt.mask_len
        self.seq_len = self.opt.isize
        self.ntoken = self.opt.ntoken
           
        self.total_steps = 0
        self.cur_epoch = 0

        #- Model
        self.G = Generator_TransformerModel(opt, self.batchsize, self.opt.ntoken, self.opt.emsize, self.opt.nhead, self.opt.nhid, self.opt.nlayer_g, self.opt.dropout)
        self.G = DataParallel(self.G)
        self.G.to(device)
        if not self.opt.istest:
            print_network(self.G)
        
        self.D = Discriminator_TransformerModel(opt, self.batchsize, self.opt.ntoken, self.opt.emsize, self.opt.nhead, self.opt.nhid, self.opt.nlayer_d, self.opt.dropout)
        self.D = DataParallel(self.D)
        self.D.to(device)
        if not self.opt.istest:
            print_network(self.D)

        #- Input & Output
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.real_label = 1
        self.fake_label= 0

        #- Loss & Optimizer
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        ##- Create mask pool
        mask_num = int(self.seq_len * self.mask_rate / self.mask_len)
        margin_length = int(self.seq_len*(1-self.mask_rate)/mask_num)

        #- base mask
        self.masks = []
        mask_base = torch.zeros_like(self.input[0,0,:])
        for i in range(0, self.seq_len, margin_length+self.mask_len):
            if i + self.mask_len > self.seq_len: break
            mask_base[i:i+self.mask_len] = torch.Tensor([self.ntoken]*(self.mask_len))
        self.masks.append(mask_base)

        #- sliding window
        window = self.mask_len//2
        shifted_mask = mask_base
        while True:
            shifted_mask = torch.roll(shifted_mask, -int(window), dims=0)
            if torch.equal(shifted_mask, mask_base): break
            self.masks.append(shifted_mask)


    def gradient_penalty(self, D, real_samples, fake_samples):
        # alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=self.device)
        real_samples = self.D.module.embedding_for_gp(real_samples)
        fake_samples = self.D.module.embedding_for_gp(fake_samples)

        alpha = torch.randn(real_samples.size(0), 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        d_interpolates = D(interpolates, gp=True)
        fake = torch.ones(real_samples.shape[0], device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def set_input(self, input):
        self.input.resize_(input[0].size()).copy_(input[0])
        self.gt.resize_(input[1].size()).copy_(input[1])
    

    def set_mask(self, input): # select mask per sample in predefined mask pool 
        self.masked_idx = torch.zeros_like(self.input)
        for bs in range(input[0].shape[0]):
            idx = random.randrange(0,len(self.masks))
            self.masked_idx[bs] = self.masks[idx]
            

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['auc'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train model.")
        start_time = time.time()
        best_auc=0
        best_auc_epoch=0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch+=1
                self.train_epoch()
                auc,th,f1=self.validate()
                self.train_hist['auc'].append(auc)

                if auc > best_auc:
                    best_auc = auc
                    best_auc_epoch=self.cur_epoch
                    self.save_weight_GD()
                    if self.opt.dataset == 'neurips_ts':
                        self.test_type_neurips_ts()
                    elif self.opt.dataset == 'mit_bih':
                        self.test_type_mit_bih()

                f.write("[{}] auc:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch, auc, best_auc, best_auc_epoch))
                print("[{}] auc:{:.4f} th:{:.4f} f1:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch, auc, th, f1, best_auc, best_auc_epoch))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        self.save(self.train_hist)
        self.save_loss(self.train_hist)
        self.save_auc(self.train_hist)


    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        self.epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize

            self.optimize(data)
            self.epoch_iter += 1

            errors = self.get_errors()
            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (self.epoch_iter  % self.opt.print_freq) == 0:
                print("Epoch: [%d] [%4d/%4d] D_loss: %.6f, G_loss: %.6f, G_rec_loss: %.6f" %
                      ((self.cur_epoch), (self.epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d"], errors["err_g"], errors["err_g_rec"]))

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)


    def optimize(self, data):
        self.forward_net(data)
        self.update_netd()
        if self.epoch_iter % 5 == 0:
            self.update_netg()


    def forward_net(self, data):

        self.set_input(data)

        # random masking (Step 1)
        self.set_mask(data)
        self.digitized, self.mult1, self.fake1, attn = self.G(self.input, self.masked_idx)

        # exclusive re-masking (Step 2)
        remasked_idx = torch.where(self.masked_idx==float(self.ntoken), 0., float(self.ntoken))

        # entropy-based re-masking (Step 2)
        attn = attn.unsqueeze(1)
        attn = torch.where(remasked_idx==0, attn.cuda(), torch.tensor(0.).cuda())
        topk_idx = torch.topk(attn, int(self.seq_len * self.mask_rate * 0.5), dim=-1)[1] # top 50%
        remasked_idx = remasked_idx.scatter(2, topk_idx.cuda(), float(self.ntoken)) # high unceratinty -> re-mask
        
        _, self.mult2, self.fake2, _ = self.G(self.input, remasked_idx)
        remasked_idx2 = remasked_idx.permute(0,2,1).repeat(1,1,self.ntoken)

        self.fake = torch.where(remasked_idx == 400, self.fake2, self.fake1)
        self.mult = torch.where(remasked_idx2 == 400, self.mult2, self.mult1)


    def update_netd(self):
        self.D.zero_grad()

        self.out_d_real = self.D(self.digitized)
        self.out_d_fake = self.D(self.fake)

        # original GAN loss
        # self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device).float())
        # self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device).float())
        # self.err_d = (self.err_d_real+self.err_d_fake)/2

        # wgan-gp loss
        gp = self.gradient_penalty(self.D, self.digitized, self.fake)
        self.err_d = -torch.mean(self.out_d_real) + torch.mean(self.out_d_fake) + self.opt.w_gp * gp

        self.err_d.backward()
        self.optimizerD.step()

        # Clip weights of discriminator
        for p in self.D.parameters():
            p.data.clamp_(-0.01, 0.01)


    def update_netg(self):
        self.G.zero_grad()
        
        self.out_g_fake = self.D(self.fake.detach())
        self.err_g_rec = self.ce_criterion(self.mult.permute(0,2,1), self.digitized.squeeze(1).long())

        # original GAN loss
        # self.err_g_adv = self.bce_criterion(self.out_g_fake, torch.full((self.batchsize,), self.real_label, device=self.device).float())
        # self.err_g = self.err_g_rec + self.err_g_adv * self.opt.w_adv

        # wgan-gp loss
        self.err_g_adv = - torch.mean(self.out_g_fake)
        self.err_g = self.err_g_rec + self.err_g_adv

        self.err_g.backward()
        self.optimizerG.step()


    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    # 'err_d_real': self.err_d_real.item(),
                    # 'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }
        
        return errors


    def validate(self):
        y_, y_pred = self.predict(self.dataloader["val"])
        rocprc, rocauc, best_th, best_f1 = evaluate(y_, y_pred)
        return rocauc, best_th, best_f1


    def predict(self,dataloader_,scale=True):
        with torch.no_grad():

            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)

                # random masking (Step 1)
                self.set_mask(data)
                self.digitized, self.mult1, self.fake1, attn = self.G(self.input, self.masked_idx)

                # exclusive re-masking (Step 2)
                remasked_idx = torch.where(self.masked_idx==float(self.ntoken), 0., float(self.ntoken))

                # entropy-based re-masking (Step2)
                attn = attn.unsqueeze(1)
                attn = torch.where(remasked_idx==0, attn.cuda(), torch.tensor(0.).cuda())
                topk_idx = torch.topk(attn, 25, dim=-1)[1] # top 50%
                remasked_idx = remasked_idx.scatter(2, topk_idx.cuda(), float(self.ntoken)) # high unceratinty -> re-mask
                
                _, self.mult2, self.fake2, _ = self.G(self.input, remasked_idx)
                self.fake = torch.where(remasked_idx == 400, self.fake2, self.fake1)

                error = torch.mean(
                    torch.pow((self.digitized.view(self.digitized.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)
                
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_ = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()

            return y_, y_pred


    def test_type_neurips_ts(self):

        self.G.eval()
        self.D.eval()
        # res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test"],scale=False)
        over_all=y_pred_N 
        over_all_gt=y_N 
        min_score,max_score=np.min(over_all),np.max(over_all)

        rocprc,rocauc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#################################")
        print("########## Test Result ##########")
        print("ap:{}".format(rocprc))
        print("auc:{}".format(rocauc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))

        return rocauc


    def test_type_mit_bih(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N=self.predict(self.dataloader["test_N"],scale=False)
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)
        over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
        over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
        min_score,max_score=np.min(over_all),np.max(over_all)
        
        aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#############################")
        print("########  Result  ###########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))