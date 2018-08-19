import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar
from modules import Encoder, Decoder              
    
    
class DeepAPI(nn.Module):
    ''' model. '''
    def __init__(self, config, vocab_size, PAD_ID=0):
        super(DeepAPI, self).__init__()
        self.vocab_size = vocab_size 
        self.maxlen=config['maxlen']
        self.clip = config['clip']
        self.temp=config['temp']
        
        self.desc_embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_ID)
        self.api_embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_ID)
                                                        # utter encoder: encode response to vector
        self.desc_encoder = Encoder(self.desc_embedder, config['emb_size'], config['n_hidden'],
                                    True, config['n_layers'], config['noise_radius']) 
        self.decoder = Decoder(self.api_embedder, config['emb_size'], config['n_hidden']*2,
                               vocab_size, config['use_attention'], 1, config['dropout']) # utter decoder: P(x|c,z)
        self.optimizer_AE = optim.Adadelta(list(self.desc_encoder.parameters())
                                      +list(self.decoder.parameters()),lr=config['lr_ae'], rho=0.95)
        self.criterion_ce = nn.CrossEntropyLoss()
    
    def train_AE(self, descs, desc_lens, apiseqs, api_lens):
        self.desc_encoder.train()
        self.decoder.train()
        
        c, hids = self.desc_encoder(descs, desc_lens)
        output = self.decoder(c, hids, None, apiseqs[:,:-1], (api_lens-1)) 
                                             # decode from z, c  # output: [batch x seq_len x n_tokens]   
        output = output.view(-1, self.vocab_size) # [batch*seq_len x n_tokens]
        
        dec_target = apiseqs[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) # 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]
        
        masked_output = output.masked_select(output_mask).view(-1, self.vocab_size)
        loss = self.criterion_ce(masked_output/self.temp, masked_target)
        
        self.optimizer_AE.zero_grad()
        loss.backward()
        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm_(list(self.desc_encoder.parameters())
                                     +list(self.decoder.parameters()), self.clip)
        self.optimizer_AE.step()
        return [('train_loss_AE', loss.item())]      
    def train_G(self, descs, desc_lens, apiseqs, api_lens):
        return []
    def train_D(self, descs, desc_lens, apiseqs, api_lens):
        return []
    
    def valid(self, descs, desc_lens, apiseqs, api_lens):
        self.desc_encoder.eval()  
        self.decoder.eval()
        
        c, hids = self.desc_encoder(descs, desc_lens)        
        dec_target = apiseqs[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]  
        output = self.decoder(c, hids, None, apiseqs[:,:-1], (api_lens-1)) 
                                     # decode from z, c  # output: [batch x seq_len x n_tokens]   
        flattened_output = output.view(-1, self.vocab_size) # [batch*seq_len x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output/self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item())]
        
    def sample(self, descs, desc_lens, n_samples, mode='beamsearch'):    
        self.desc_encoder.eval()
        self.decoder.eval()
        c, hids = self.desc_encoder(descs, desc_lens)
        if mode =='beamsearch':
            sample_words, sample_lens, _ = self.decoder.beam_search(c, hids, None, n_samples, self.maxlen)
                                                                   #[batch_size x n_samples x seq_len]
            sample_words, sample_lens = sample_words[0], sample_lens[0]
        else:
            sample_words, sample_lens = self.decoder.sampling(c, hids, None, n_samples, self.maxlen, mode) 
        return sample_words, sample_lens   
    
    def adjust_lr(self):
        #self.lr_scheduler_AE.step()
        return None
    
    
    

