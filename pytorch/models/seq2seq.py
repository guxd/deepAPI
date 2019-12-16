import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import PAD_ID
from modules import CharEmbedding, RNNEncoder, RNNDecoder, ScheduledOptim             
    
    
class RNNSeq2Seq(nn.Module):
    '''The basic Hierarchical Recurrent Encoder-Decoder model. '''
    def __init__(self, config):
        super(RNNSeq2Seq, self).__init__()
        self.vocab_size = config['vocab_size'] 
        self.maxlen=config['max_sent_len']
        self.clip = config['clip']
        self.temp=config['temp']
        
        self.desc_embedder = nn.Embedding(self.vocab_size, config['emb_dim'], padding_idx=PAD_ID)
        self.api_embedder = nn.Embedding(self.vocab_size, config['emb_dim'], padding_idx=PAD_ID)
                                                        
        self.encoder = RNNEncoder(self.desc_embedder, None, config['emb_dim'], config['n_hidden'],
                        True, config['n_layers'], config['noise_radius']) # utter encoder: encode response to vector
        self.ctx2dec = nn.Sequential( # from context to decoder initial hidden
            nn.Linear(2*config['n_hidden'], config['n_hidden']),
            nn.Tanh(),
        )
        self.ctx2dec.apply(self.init_weights)
        self.decoder = RNNDecoder(self.api_embedder, config['emb_dim'], config['n_hidden'],
                               self.vocab_size, config['attention'], 1, config['dropout']) # decoder: P(x|c,z)
        
        self.optimizer = ScheduledOptim(optim.Adam(
            filter(lambda x: x.requires_grad, self.parameters()),
            betas=(0.9, 0.98), eps=1e-09), config['n_hidden'], config['n_warmup_steps'])
        
        self.criterion_ce = nn.CrossEntropyLoss()
        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)   
        
    def forward(self, src_seqs, src_lens, target, tar_lens):
        c, hids = self.encoder(src_seqs, src_lens)
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids)
        src_pad_mask = src_seqs.eq(PAD_ID)
        output,_ = self.decoder(init_h, hids, src_pad_mask, None, target[:,:-1], (tar_lens-1)) 
                                             # decode from z, c  # output: [batch x seq_len x n_tokens]   
        output = output.view(-1, self.vocab_size) # [batch*seq_len x n_tokens]
        
        dec_target = target[:,1:].contiguous().view(-1)
        mask = dec_target.gt(PAD_ID) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) # 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]
        
        masked_output = output.masked_select(output_mask).view(-1, self.vocab_size)
        loss = self.criterion_ce(masked_output/self.temp, masked_target)
        return loss

    def train_AE(self, src_seqs, src_lens, target, tar_lens):
        self.train()   
        
        loss=self.forward(src_seqs, src_lens, target, tar_lens)
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self.optimizer.step()
        return {'train_loss': loss.item()}   

    def valid(self, src_seqs, src_lens, target, tar_lens):
        self.eval()       
        loss = self.forward(src_seqs, src_lens, target, tar_lens)
        return {'valid_loss': loss.item()}
        
    def sample(self, src_seqs, src_lens, n_samples, decode_mode='beamsearch'):    
        self.eval()
        src_pad_mask = src_seqs.eq(PAD_ID)
        c, hids = self.encoder(src_seqs, src_lens)
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids)
        if decode_mode =='beamsearch':
            sample_words, sample_lens, _ = self.decoder.beam_decode(init_h, hids, src_pad_mask, None, 12, self.maxlen, n_samples)
                                                                   #[batch_size x n_samples x seq_len]
            sample_words, sample_lens = sample_words[0], sample_lens[0]
        else:
            sample_words, sample_lens = self.decoder.sampling(init_h, hids, src_pad_mask, None, self.maxlen, decode_mode)  
        return sample_words, sample_lens   
    
    def adjust_lr(self):
        #self.lr_scheduler_AE.step()
        return None
    
    

