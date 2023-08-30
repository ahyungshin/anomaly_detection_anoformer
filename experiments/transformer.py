import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

from utils import SoftArgmax2D
from transformer_util import TransformerEncoder, TransformerEncoderLayer


class Generator_TransformerModel(nn.Module):
    def __init__(self, opt, bs, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Generator_TransformerModel, self).__init__()
        self.ntoken=ntoken
        self.ninp = ninp
        self.nhead = nhead
        self.ngpu = opt.ngpu

        #- embed
        self.embedding = Embedding(bs, ntoken, ninp, dropout)
        #- transformer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #- inverse embed
        self.inverse_embedding = InverseEmbedding(ntoken) 

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.embedding.weight.data.uniform_(-initrange, initrange)
        
    def digitize(self, input):
       
        max_input, min_input = 1, -1 #normalized
        split = (max_input - min_input) / self.ntoken 
        
        bin=[]
        for i in range(self.ntoken):
            min_input += split
            bin.append(min_input)
        bin = np.array(bin)
        device = input.device
        input = input.detach().cpu().numpy()
        output = np.digitize(input, bins=bin).squeeze(axis=1) #[-1,1] -> [0,399]

        output = bin[output]
        output += (-bin[0])
        output *= (self.ntoken/2)

        output = torch.from_numpy(output).to(device).to(dtype=torch.float) 
        return output


    def mask(self, input, idxs):
        input = input.unsqueeze(1)
        output = input.masked_fill(idxs==self.ntoken, self.ntoken)

        return output.squeeze(1)


    def forward(self, src, masked_idx=None, digitize=False):
        if digitize is False:
            #- quantization
            src = self.digitize(src) 
        else:
            src = src.squeeze(1)
        #- mask
        mask =  self.mask(src, masked_idx).to(dtype=torch.long)
        #- embed
        embed, embed_weight = self.embedding(mask) 
        # Transformer
        tf = embed.permute(2,0,1) 
        tf, attn = self.transformer_encoder(tf, tf, tf) 
        tf_output = tf.permute(1,2,0) 

        # Inverse Embedding
        mult, fake = self.inverse_embedding(tf_output, embed_weight) 
        fake = fake.to(dtype=torch.float)
        return src.unsqueeze(1), mult, fake, attn



class Discriminator_TransformerModel(nn.Module):
    def __init__(self, opt, bs, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Discriminator_TransformerModel, self).__init__()
        
        self.bs = bs
        self.ntoken=ntoken
        self.ninp = ninp
        self.nhead = nhead
        self.ngpu = opt.ngpu

        #- embed
        self.embedding = Embedding(bs, ntoken, ninp, dropout)
        #- transformer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #- classifier
        self.linear = nn.Linear(self.ninp, 1)
        self.layer_norm = m = nn.LayerNorm(self.ninp)
        self.sigmoid = nn.Sigmoid()

        self.cls_token = nn.Parameter(torch.randn(self.ninp))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.uniform_(0,0)

    def embedding_for_gp(self, src):
        embed, _ = self.embedding(src.to(dtype=torch.long).squeeze(1)) #[B,L] -> [B,D,L]
        cls_token = self.cls_token.unsqueeze(0).unsqueeze(2).repeat(src.shape[0],1,1) # [B,D,1]
        src = torch.cat([cls_token, embed], dim=2) # [B,D,L+1]

        return src


    def forward(self, src, gp=False):

        if gp is False:
            embed, _ = self.embedding(src.to(dtype=torch.long).squeeze(1)) #[B,L] -> [B,D,L]
            cls_token = self.cls_token.unsqueeze(0).unsqueeze(2).repeat(src.shape[0],1,1) # [B,D,1]
            src = torch.cat([cls_token, embed], dim=2) # [B,D,L+1]

        else:
            embed = src

        tf = embed.permute(2,0,1)
        tf, attn = self.transformer_encoder(tf)  
        tf = tf.permute(1,2,0) 

        cls_embed = tf[:,:,0].unsqueeze(2).permute(0,2,1) 
        classifier = self.linear(cls_embed)
        classifier = self.sigmoid(classifier)
        classifier = classifier.view(-1,1).squeeze(1)

        return classifier


class Embedding(nn.Module):
    def __init__(self, bs, ntoken, ninp, dropout):
        super(Embedding, self).__init__()

        self.ntoken = ntoken
        self.ninp = ninp

        self.embedding = nn.Embedding(ntoken+1, ninp) # +1:  [mask] in G / [cls] in D
        self.pos_encoder = PositionalEncoding(bs, ninp, dropout)


    def forward(self, src): 
        embed = self.embedding(src) * math.sqrt(self.ninp) 
        embed = self.pos_encoder(embed) 
        embed = embed.permute(0,2,1) 
        return embed, self.embedding.weight.clone().detach().transpose(0,1)


class PositionalEncoding(nn.Module):

    def __init__(self, bs, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.stack([pe]*bs, dim=0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1), :] 
        return self.dropout(x)


class InverseEmbedding(nn.Module):
    def __init__(self, ntoken):
        super(InverseEmbedding, self).__init__()
        self.ntoken = ntoken
        self.argmax = SoftArgmax2D()

    def forward(self, input, weight):
        #normalize
        input = input.permute(0,2,1) 
        input_abs = torch.norm(input, dim=2)
        weight_abs = torch.norm(weight, dim=0) 

        input = input.permute(2,0,1) 
        input_norm = torch.div(input, input_abs) 
        weight_norm = torch.div(weight, weight_abs)
      
        input_norm = input_norm.permute(1,2,0)
        mult = torch.matmul(input_norm, weight_norm[:,:self.ntoken]) 
        output = self.argmax(mult) 
        output = output.permute(0,2,1) 
        return mult, output