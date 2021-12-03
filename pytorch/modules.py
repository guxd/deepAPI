import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import math
import random
from queue import PriorityQueue
import operator
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import get_position, SOS_ID, EOS_ID, PAD_ID


class MLP(nn.Module):
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_weight=0.02):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_weight= init_weight

        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = self.init_weight
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
            
class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    """
    def __init__(self, input_dim, n_layers=1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(n_layers)])
        self.activation = nn.ReLU()        
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            # As per comment in AllenNLP: We should bias the highway layer to just carry its input forward.  
            # We do that by setting the bias on`B(x)`to be positive, because that means`g`will be biased to be high,
            # so we will carry the input forward. The bias on`B(x)`is the second half of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * proj_x
        return x
    
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None  
    
class CharEmbedding(nn.Module):
    '''
     adopted from https://github.com/seongjunyun/Character-Aware-Neural-Language-Models/blob/master/model.py
     and bidaf: https://github.com/jojonki/BiDAF/blob/master/layers/char_embedding.py
     In : (batch_size, sent_len, word_len)
     Out: (batch_size, sent_len, c_emb_size)
     '''
    def __init__(self, c_vocab_size, c_emb_dim, max_word_len, out_channels=25, kernels = [1,2,3,4,5,6]):
        super(CharEmbedding, self).__init__()
        self.emb_dim = c_emb_dim
        self.embedding = nn.Embedding(c_vocab_size, c_emb_dim)        
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, out_channels*kernel_width, kernel_size=(kernel_width, c_emb_dim)), # feature map
                nn.Tanh(),
                nn.MaxPool2d((max_word_len-kernel_width+1,1))
            ) for kernel_width in kernels]) 
        cnn_out_size = np.asscalar(out_channels*np.sum(kernels))
        self.highway = Highway(cnn_out_size, n_layers = 2)
        self.batch_norm = nn.BatchNorm1d(cnn_out_size, affine=False)
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(cnn_out_size, c_emb_dim)
        
        self.init_weights()
        
    def init_weights(self):
        #self.embed.weight.data.uniform_(-0.05,0.05)
        for conv in self.convolutions:
            conv[0].weight.data.uniform_(-0.05,0.05)
            conv[0].bias.data.fill_(0)       
        self.highway.init_weights()
        self.fc.weight.data.uniform_(-0.05,0.05)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_len, word_len = x.size()
        x = x.view(-1, word_len) # [(batch_size*seq_len) x word_len]
        x = self.embedding(x) # [(batch_size*seq_len) x word_len x c_emb_dim]
        x = x.unsqueeze(1)# [(batch_size*seq_len) x 1 x  word_len x c_embd_dim] insert channel-in dim 
        y = [conv(x).view(batch_size*seq_len, -1) for conv in self.convolutions]     
        x = torch.cat(y,1)# [(batch_size*seq_len) x cnn_out_size]
        x = self.batch_norm(x)
        x = self.highway(x)
        x = x.contiguous().view(batch_size,seq_len, -1) # [batch_size, seq_len, cnn_out_size]
        x = self.dropout(x)  
        x = self.fc(x) # project back to char embed dim [batch_size, seq_len, c_emb_dim]
        return x
    """
    How to combine char embed with word embed:
        char_embd = self.char_embd_net(x_c) # (N, seq_len, embd_size)
        word_embd = self.word_embd_net(x_w) # (N, seq_len, embd_size)
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, d=embd_size*2)
        embd = self.highway_net(embd) # (N, seq_len, d=embd_size*2)
    """
    
###############################################################################################################################    
            
class RNNEncoder(nn.Module):
    def __init__(self, embedder, char_embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
        super(RNNEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.word_embedder = embedder # nn.Embedding(vocab_size, emb_size)
        if embedder is not None: self.emb_dim = embedder.weight.size(1)
        self.char_embedder = char_embedder
        if char_embedder is not None: 
            self.highway = Highway(self.emb_dim+char_embedder.emb_dim)
            self.comb_emb = nn.Linear(self.emb_dim+char_embedder.emb_dim, self.emb_dim)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_weights()
        
    def init_weights(self):
        """ adopted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5 """
        for name, param in self.rnn.named_parameters(): # initialize the gate weights 
            # adopted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
            #if len(param.shape)>1:
            #    weight_init.orthogonal_(param.data) 
            #else:
            #    weight_init.normal_(param.data)                
            # adopted from fairseq
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)
    
    def forward(self, inputs, char_inputs=None, input_lens=None, init_h=None, noise=False): 
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.word_embedder is not None:
            inputs=self.word_embedder(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        if self.char_embedder is not None:
            char_emb = self.char_embedder(char_inputs)
            word_emb = torch.cat((inputs, char_emb), 2)            
            word_emb = self.highway(word_emb) # use a highway to combine word emb with char emb
            inputs = self.comb_emb(word_emb)
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
        
        if init_h is None:
            init_h = self.init_h.expand(-1, batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        #self.rnn.flatten_parameters() # time consuming!!
        hids, (h_n, c_n) = self.rnn(inputs) # lstm
        #hids, h_n = self.gru(inputs, init_h) # hids: [b x seq x (n_dir*hid_sz)]  
                                 # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)  
            hids = F.dropout(hids, p=0.25, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
############commenting the following line significantly improves the performance, why? #####################################
        #h_n = h_n.transpose(1,0).contiguous()
        #enc = h_n.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        enc = torch.cat((h_n[0], h_n[1]), dim=1)
        
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(enc.size(), device=inputs.device),std=self.noise_radius)
            enc = enc + gauss_noise
            
        return enc, hids  
    
    
class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the hidden features from the decoder.
    Adopted from https://github.com/snakeztc/NeuralDialog-ZSDG/blob/master/zsdg/enc2dec/decoders.py
    Args:
    Inputs: Q, K, V
        - **Q** (batch, trg_len, dimensions): tensor containing the query, i.e., output states from the decoder.
        - **K** (batch, src_len, dimensions): tensor containing the keys, i.e., hiddens of the encoded input sequence.
        - **V** =K
    Outputs: output, attn
        - **mix** (batch, trg_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, trg_len, src_len): tensor containing attention weights.
    Attributes:
        attn_combine (torch.nn.Linear): return a linear combination of querys and values.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    """
    def __init__(self, q_dim, k_dim, v_dim, mode='concat', dropout = 0., attn_combine=False):
        """
        :param attn_combine: combine weighted query with query using a fully connected
        """
        super(Attention, self).__init__()
        self.mode = mode
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.dropout = nn.Dropout(dropout)
        
        if mode == 'dot':
            self.temperature = np.sqrt(self.k_dim) # a scale factor for softmax, used in transformer
        if mode == 'general':
            self.w_q2k = nn.Linear(q_dim, k_dim)
        elif mode == 'concat':
            self.w_q = nn.Linear(q_dim, q_dim)
            self.w_k = nn.Linear(k_dim, q_dim)
            self.w_V = nn.Linear(q_dim, 1)
        self.attn_combine = attn_combine 
        if attn_combine:
            self.linear_comb = nn.Linear(q_dim+v_dim, q_dim)
            
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.1, 0.1)
            if 'bias' in name:
                nn.init.constant_(param, 0.)
            
    def forward(self, Q, K, V, key_pad_mask=None, attn_mask=None):
        """
        :param Q: [batch, q_len, q_dim]
        :param K: [batch, k_len, k_dim]
        :param V: [batch, v_len, v_dim]
        :param key_pad_mask: [batch_size, k_len], v_ij == 1 indicates that key(src) token j is a pad token in batch i.
        :param attn_mask: [q_len, k_len], v_ij == -inf indicates that position j in the key(src) should not be attended by position i in the query (tgt).
        :return: output, attn
        """
        batch_size, q_len, _ = Q.size()
        k_len = K.size(1)

        if self.mode == 'dot': # Scaled Dot Product Attention e_i = K^TQ 
            attn_scores = torch.bmm(Q, K.transpose(1, 2))
            attn_scores = attn_scores/self.temperature
        elif self.mode == 'general': # e_i = Q^TWK
            mapped_Q = self.w_q2k(Q)
            attn_scores = torch.bmm(mapped_Q, K.transpose(1, 2))
        elif self.mode == 'concat': # e_i = V^T tanh(W_kK+W_qQ)
            mapped_K = self.w_k(K)
            mapped_Q = self.w_q(Q)
            tiled_Q = mapped_Q.unsqueeze(2).repeat(1, 1, k_len, 1)
            tiled_K = mapped_K.unsqueeze(1)
            fc1 = torch.tanh(tiled_K+tiled_Q)
            attn_scores = self.w_V(fc1).squeeze(-1)
        else:
            raise ValueError("Unknown attention")
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0)
        if key_pad_mask is not None:
            attn_scores = attn_scores.masked_fill_(key_pad_mask.unsqueeze(1), -1e12)

        attn_weights = F.softmax(attn_scores.view(-1, k_len), dim=1).view(batch_size, -1, k_len) # (batch x q_len x k_len)
        attn_weights = self.dropout(attn_weights)
        attn_ctx = torch.bmm(attn_weights, V) # (batch, q_len, k_len) * (batch, k_len, v_dim)  -> (batch, q_len, v_dim)
        
        combined_attn_ctx = None # combine attended context with query
        if self.attn_combine:
            combined = torch.cat((Q, attn_ctx), dim=2)
            combined_attn_ctx = torch.tanh(self.linear_comb(combined.view(-1, self.q_dim+self.v_dim))).view(batch_size,-1,self.q_dim)
        return attn_ctx, attn_weights, combined_attn_ctx # attn [batch_size x q_len x k_len]
    

class RNNDecoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, use_attention=False, n_layers=1, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size # size of the input to the RNN (e.g., embedding dim)
        self.hidden_size = hidden_size # RNN hidden size
        self.vocab_size = vocab_size # RNN output size (vocab size)
        self.dropout= dropout
        
        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.project = nn.Linear(hidden_size, vocab_size)
        
        self.use_attention = use_attention
        if use_attention:
            self.attn = Attention(self.hidden_size, self.hidden_size, self.hidden_size, attn_combine=False)
            self.x_context = nn.Linear(hidden_size + input_size, input_size) # combine input with attention states
            self.rnn = nn.GRU(input_size + hidden_size, hidden_size, batch_first=True)
            self.project = nn.Linear(2*self.hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.project.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(self.out.weight)        
        nn.init.constant_(self.project.bias, 0.)
        if self.use_attention:
            self.x_context.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(self.out.weight)        
            nn.init.constant_(self.x_context.bias, 0.)
    
    def forward(self, init_h, enc_hids, src_pad_mask, context=None, inputs=None, lens=None):
        '''
        init_h: initial hidden state for decoder
        enc_hids: enc_hids for attention use
        context: context information to be paired with input
        inputs: inputs to the decoder
        lens: input lengths
        '''
        if self.embedding is not None:
            inputs = self.embedding(inputs) # input: [batch_sz x seqlen x emb_sz]
        batch_size, maxlen, _ = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)  
        h = init_h.unsqueeze(0) # last_hidden of decoder [n_dir x batch_sz x hid_sz]        
        if self.use_attention:  
            attn_ctx, _, _ = self.attn(init_h.unsqueeze(1), enc_hids, enc_hids, src_pad_mask)# initial attention context
            for di in range(maxlen):
                x = inputs[:,di,:].unsqueeze(1) # [batch_sz x 1 x emb_sz]
                
                x = torch.cat((x, attn_ctx),2) # [batch_sz x 1 x (emb_sz+hid_sz)]
                #x = self.x_context(torch.cat((x, attn_ctx), 2))
                
                h_n, h = self.rnn(x, h) # h_n: [batch_sz x 1 x hid_sz] h: [1 x batch_sz x hid_sz]
                attn_ctx, _, comb_attn_ctx = self.attn(h_n, enc_hids, enc_hids, src_pad_mask) 
                                  # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                    
                logits = self.project(torch.cat((h_n, attn_ctx),2)) # out: [batch_sz x 1 x vocab_sz]  
                #logits=self.project(comb_attn_ctx) # out: [batch_sz x 1 x vocab_sz]        
                
                decoded = logits if di==0 else torch.cat([decoded, logits],1) # decoded: [batch_sz x maxlen x vocab_sz]
        else:
            if context is not None:            
                repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1) # [batch_sz x max_len x hid_sz]
                inputs = torch.cat([inputs, repeated_context], 2)
                
            #self.rnn.flatten_parameters()
            hids, h = self.rnn(inputs, h)         
            decoded = self.project(hids.contiguous().view(-1, self.hidden_size))# reshape before linear over vocab
            decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded, h
    
    def sampling(self, init_h, enc_hids, src_pad_mask, context, maxlen, mode='greedy', to_numpy=True):
        """
        A simple greedy sampling
        :param init_h: [batch_sz x hid_sz]
        :param enc_hids: a tuple of (enc_hids, mask) for attention use. [batch_sz x seq_len x hid_sz]
        """
        device = init_h.device
        batch_size = init_h.size(0)
        decoded_words = torch.zeros((batch_size, maxlen), dtype=torch.long, device=device)  
        sample_lens = torch.zeros((batch_size), dtype=torch.long, device=device)
        len_inc = torch.ones((batch_size), dtype=torch.long, device=device)
               
        x = torch.zeros((batch_size, 1), dtype=torch.long, device=device).fill_(SOS_ID)# [batch_sz x 1] (1=seq_len)
        h = init_h.unsqueeze(0) # [1 x batch_sz x hid_sz]  
        if self.use_attention:  
            attn_ctx, _, _ = self.attn(init_h.unsqueeze(1), enc_hids, enc_hids, src_pad_mask)# initial attention context
        for di in range(maxlen):  
            if self.embedding is not None:
                x = self.embedding(x) # x: [batch_sz x 1 x emb_sz]
            x = torch.cat((x, attn_ctx),2) # [batch_sz x 1 x (emb_sz+hid_sz)]
            #x = self.x_context(torch.cat((x, attn_ctx), 2))
                
            h_n, h = self.rnn(x, h) # h_n: [batch_sz x 1 x hid_sz] h: [1 x batch_sz x hid_sz]
            attn_ctx, _, comb_attn_ctx = self.attn(h_n, enc_hids, enc_hids, src_pad_mask) 
                                 # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                    
            logits = self.project(torch.cat((h_n, attn_ctx),2)) # out: [batch_sz x 1 x vocab_sz]  
            logits = logits.squeeze(1) # [batch_size x vocab_size]   
            if mode=='greedy':
                x = logits.max(1, keepdim=True)[1] # x:[batch_sz x 1] indexes of predicted words
            elif mode=='sample':                
                x = torch.multinomial(F.softmax(logits, dim=1), 1)  # [batch_size x 1 x 1]?
            decoded_words[:,di] = x.squeeze()
            len_inc=len_inc*(x.squeeze()!=EOS_ID).long() # stop increse length (set 0 bit) when EOS is met
            sample_lens=sample_lens+len_inc            
        
        if to_numpy:
            decoded_words = decoded_words.data.cpu().numpy()
            sample_lens = sample_lens.data.cpu().numpy()
        return decoded_words, sample_lens
    
    def beam_decode(self, init_h, enc_hids, src_pad_mask, context, beam_width, max_unroll, topk=1):
        '''
        https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        :param init_h: input tensor of shape [B, H] for start of the decoding
        :param enc_hids: if you are using attention mechanism you can pass encoder outputs, [B, T, H] where T is the maximum length of input sentence
        :param topk: how many sentence do you want to generate
        :return: decoded_batch
        '''      
        device=init_h.device
        batch_size=init_h.size(0)
        decoded_words = np.zeros((batch_size, topk, max_unroll), dtype=np.int)
        sample_lens =np.zeros((batch_size, topk), dtype=np.int)
        scores = np.zeros((batch_size, topk))
        
        for idx in range(batch_size): # decoding goes sentence by sentence
            if isinstance(init_h, tuple):  # LSTM case
                h = (init_h[0][idx,:].view(1,1,-1),init_h[1][idx,:].view(1,1,-1))
            else:
                h = init_h[idx,:].view(1,1,-1)            
            if enc_hids is not None:
                enc_outs = enc_hids
                enc_outs = enc_outs[idx,:,:].unsqueeze(0) 
                src_pad_mask = src_pad_mask[idx, :].unsqueeze(0)

            # Start with the start of the sentence token
            x = torch.zeros((1, 1), dtype=torch.long, device=device)
            if self.use_attention:  
                attn_ctx, _, _ = self.attn(h, enc_hids, enc_hids, src_pad_mask)# initial attention context
                
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk-len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamNode(h, None, x, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:                
                if qsize > 2000: break # give up when decoding takes too long
                    
                score, n = nodes.get() # fetch the best node
                x = n.wordid
                h = n.h
                qsize-=1

                if n.wordid.item() == EOS_ID and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                if self.embedding is not None:
                    x = self.embedding(x) # x: [1 x 1 x emb_sz]
                x = torch.cat((x, attn_ctx),2) # [1 x 1 x (emb_sz+hid_sz)]
                
                h_n, h = self.rnn(x, h) # h_n: [1 x 1 x hid_sz] h: [1 x batch_sz x hid_sz]
                attn_ctx, _, comb_attn_ctx = self.attn(h_n, enc_hids, enc_hids, src_pad_mask) 
                                 # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                    
                logits = self.project(torch.cat((h_n, attn_ctx),2)) # out: [1x 1 x vocab_sz]  
                logits = logits.squeeze(1) # [1 x vocab_size]  
                logits = F.log_softmax(logits, 1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(logits, beam_width) # [1 x beam_width]

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()
                    node = BeamNode(h, n, decoded_t, n.logp + log_p, n.len + 1)
                    score = -node.eval()
                    nodes.put((score, node))# put them into queue
                    qsize += 1 # increase qsize

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            uid=0
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance, length = [], n.len
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)
                utterance = utterance[::-1] #reverse
                utterance, length = utterance[1:], length-1 # remove <sos>
                decoded_words[idx,uid,:min(length, max_unroll)]=utterance[:min(length, max_unroll)]
                sample_lens[idx,uid]=min(length, max_unroll)
                scores[idx,uid]=score
                uid=uid+1
                
        return decoded_words, sample_lens, scores   

class BeamNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb # log prob from start to the node
        self.len = length # length up to the node

    def eval(self, alpha=1.0):
        reward = 0.
        # Add here a function for shaping a reward
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward # average logp up to the current node

    def __lt__(self, other):
        '''overwrith lt function. when two nodes have the same score, the priority queue will compare the two nodes themselves.'''
        print("[warning] two nodes have the same score (%d-%d), possible recursive (cycle) search."%(self.wordid, other.wordid)) 
        return self.len>=other.len 
    
 
     
#########################################################################################################################    
    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)  