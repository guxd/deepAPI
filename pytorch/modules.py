import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar, gData, sequence_mask, SOS_ID, EOS_ID
            
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
    
class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2, dropout=0.5):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        assert type(self.bidirectional)==bool
        self.dropout = dropout
        
        self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    
    def forward(self, inputs, input_lens=None, noise=False): 
        if self.embedding is not None:
            inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        #self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs) # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            hids = (hids, sequence_mask(input_lens)) # append mask for attention
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidirectional), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = h_n.transpose(1,0).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()),std=self.noise_radius))
            enc = enc + gauss_noise
            
        return enc, hids
    

class AttentionPooling(nn.Module):
    def __init__(self, key_size, query_sizes, attn_size=75, gamma=1.0):
        super().__init__()
        self.key_linear = nn.Linear(key_size, attn_size, bias=False)
        self.query_linears = nn.ModuleList([nn.Linear(query_size, attn_size, bias=False) for query_size in query_sizes])
        self.score_linear = nn.Linear(attn_size, 1, bias=False)
        self.gamma = gamma

    def _calculate_scores(self, source, source_mask, score_unnormalized):
        """
        :return: if batch_first: seq x batch x n   context vectors
        """
        if source_mask is not None:
            score_unnormalized.data.masked_fill_(source_mask.data != 1, -1e12)
        n_batch, seq_len, _ = source.size()
        scores = (F.softmax(score_unnormalized.view(-1, seq_len)/self.gamma, 1)
                  .view(n_batch, seq_len, 1))
        return scores

    def forward(self, key, queries, key_mask=None, values=None):
        """
        :param key: [batch_sz x seq_len x hid_sz]
        :param queries: a list of query  [[batch_sz x 1 x hid_sz]]
        :param key_mask: [batch_sz x seq_len x 1]
        :param values: [batch_sz x seq_len x hid_sz]
        :return:
        """
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(2)
        if values is None:
            values = key

        score_before_transformation = self.key_linear(key)

        for i, query in enumerate(queries):
            score_before_transformation = score_before_transformation + self.query_linears[i](query)
        score_unnormalized = self.score_linear(F.tanh(score_before_transformation))
        scores = self._calculate_scores(key, key_mask, score_unnormalized)
        context = torch.bmm(scores.transpose(1, 2), values)
        return scores, context
    
class Decoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, use_attention=False, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size # size of the input to the RNN (e.g., embedding dim)
        self.hidden_size = hidden_size # RNN hidden size
        self.vocab_size = vocab_size # RNN output size (vocab size)
        self.dropout = dropout

        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        
        self.use_attention = use_attention
        if use_attention:
            self.rnn = nn.GRU(input_size + hidden_size, hidden_size, batch_first=True)
            self.attn = AttentionPooling(self.hidden_size, [self.hidden_size], self.hidden_size)
            self.out = nn.Linear(2*self.hidden_size, vocab_size) 
            
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)# Initialize Linear Weight
        self.out.bias.data.fill_(0)
    
    def forward(self, init_h, enc_hids, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs) # input: [batch_sz x seqlen x emb_sz]
        
        inputs = F.dropout(inputs, self.dropout, self.training) 
        
        h = init_h.unsqueeze(0) # last_hidden of decoder [n_dir x batch_sz x hid_sz]
        enc_hids, enc_hids_mask = enc_hids
        if self.use_attention:            
            attn_ctx = gVar(torch.zeros(batch_size,1, self.hidden_size))
            for di in range(inputs.size(1)):
                x = inputs[:,di,:].unsqueeze(1) # [batch_sz x 1 x emb_sz]
                x = torch.cat((x, attn_ctx),2) # [batch_sz x 1 x (emb_sz+hid_sz)]
                h_n, h = self.rnn(x, h) # h_n: [batch_sz x 1 x hid_sz] h: [1 x batch_sz x hid_sz]
                queries=[h.transpose(0,1)] # [[batch_sz x 1 x hid_sz]]
                attn_weights, attn_ctx = self.attn(enc_hids, queries, enc_hids_mask) 
                                  # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                out=self.out(torch.cat((h_n, attn_ctx),2)) # out: [batch_sz x 1 x vocab_sz]
                decoded = out if di==0 else torch.cat([decoded, out],1) # decoded: [batch_sz x maxlen x vocab_sz]
        else:
            if context is not None:            
                repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1) # [batch_sz x max_len x hid_sz]
                inputs = torch.cat([inputs, repeated_context], 2)                
            #self.rnn.flatten_parameters()
            hids, h = self.rnn(inputs, h)         
            decoded = self.out(hids.contiguous().view(-1, self.hidden_size))# reshape before linear over vocab
            decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded
    
    def sampling(self, init_h, enc_hids, context,  maxlen, mode='greedy'):
        """
        A simple greedy sampling
        :param init_h: [batch_sz x hid_sz]
        :param encoder_hids: [batch_sz x seq_len x hid_sz] encoder hiddens for attention
        """
        batch_size=init_h.size(0)
        decoded_words = np.zeros((batch_size, maxlen), dtype=np.int)
        sample_lens=np.zeros(batch_size, dtype=np.int)         
    
        x = gVar(torch.LongTensor([[SOS_ID]*batch_size]).view(batch_size,1))# [batch_sz x 1] (1=seq_len)
        x = self.embedding(x) if self.embedding is not None else x #[batch_sz x seqlen x emb_sz]
        
        h = init_h.unsqueeze(0) # [1 x batch_sz x hid_sz]   
        enc_hids, enc_hids_mask = enc_hids
        attn_ctx = gVar(torch.zeros(batch_size,1, self.hidden_size))
        
        for di in range(maxlen):
            if self.use_attention:
                x = torch.cat((x, attn_ctx),2) # [batch_sz x 1 x hid_sz]
                h_n, h = self.rnn(x, h) # h_n: [batch_sz x 1 x hid_sz] h: [1 x batch_sz x hid_sz]
                queries=[h.transpose(0,1)] # [[batch_sz x 1 x hid_sz]]
                attn_weights, attn_ctx = self.attn(enc_hids, queries, enc_hids_mask) 
                                  # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                out=self.out(torch.cat((h_n, attn_ctx),2)) # out: [batch_sz x 1 x vocab_sz]
                
            else:
                x = torch.cat([x, context.unsqueeze(1)],2) if context is not None else x
                h_n, h = self.rnn(x, h) #decoder_output:[batch_sz x 1 x vocab_sz]
                out=self.out(h_n)# out: [batch_sz x 1 x vocab_sz]
                
            if mode=='greedy':
                topi = decoder_output[:,-1].max(1, keepdim=True)[1] # topi:[batch_sz x 1] indexes of predicted words
            elif mode=='sample':
                topi = torch.multinomial(F.softmax(decoder_output[:,-1], dim=1), 1)                    
            x = self.embedding(topi) if self.embedding is not None else topi
            decoded_words[:,di] = topi.squeeze().data.cpu().numpy() #!!
                    
        for i in range(batch_size):
            for word in decoded_words[i]:
                if word == EOS_ID:
                    break
                sample_lens[i]=sample_lens[i]+1
        return decoded_words, sample_lens  
    
    def beam_search(self, init_h, enc_hids, context, beam_size, max_unroll):
        """
        Args:
            init_h (variable, FloatTensor): [batch_size x hid_size]
            encoder_hids: [batch_size x seq_len x hid_size]
        Return:
            out: [batch_size, seq_len]
        """
        batch_size = init_h.size(0)
        x = gVar(torch.LongTensor([SOS_ID] *batch_size*beam_size))# [batch_size*beam_size]
        if self.use_attention:             
            enc_hids, enc_hids_mask = enc_hids
            enc_hids = enc_hids.repeat(beam_size, 1, 1) # repeat encoder hiddens by beam_size times
            enc_hids_mask = enc_hids_mask.repeat(beam_size, 1) 
        h = init_h.unsqueeze(0).repeat(1,beam_size,1).contiguous() # [n_layers=1 x (batch_size*beam_size) x hid_size]
            # ?? tile or repeat?
            
        batch_position = gVar(torch.arange(0, batch_size).long() * beam_size) #[batch_size]  
           #Points where batch starts in [batch_size x beam_size] tensors 
           # [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]  Eg. position_idx[5]: when 5-th batch starts

        # Initialize scores of sequence [(batch_size*beam_size)]
        # Ex. batch_size: 5, beam_size: 3  [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        score = torch.ones(batch_size*beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * beam_size, 0.0)
        score = gVar(score)
        
        beam = Beam(batch_size, beam_size, max_unroll, batch_position) # Initialize Beam that stores decisions for backtracking
        
        attn_ctx = gVar(torch.zeros(batch_size*beam_size, 1, self.hidden_size))
        
        for i in range(max_unroll):
            x = x.view(-1, 1) # [(batch_size*beam_size)] => [(batch_size*beam_size) x 1]
            x = self.embedding(x) if self.embedding is not None else x # [(batch_size*beam_size) x 1 x emb_sz]
            if self.use_attention:
                x = torch.cat((x, attn_ctx),2) # [batch_sz x 1 x hid_sz]
                h_n, h = self.rnn(x, h)# h: [n_layers=1 x (batch_size*beam_size) x hidden_size]
                queries=[h[0].unsqueeze(1)] # [[(batch_sz*beam_sz) x 1 x hid_sz]]
                attn_weights, attn_ctx = self.attn(enc_hids, queries, enc_hids_mask) 
                                  # attn_ctx: [batch_sz x 1 x hid_sz] weights: [batch_sz x seq_len x 1]
                out=self.out(torch.cat((h_n, attn_ctx),2)).squeeze(1) # out: [(batch_sz*beam_sz) x vocab_sz]
            else:
                x = torch.cat([x, context.unsqueeze(1)],2) if context is not None else x
                out, h = self.rnn(x, h)# h: [n_layers=1 x (batch_size*beam_size) x hidden_size]
                out = self.out(out).squeeze(1)    # [(batch_size*beam_size) x vocab_size]    
            
            log_prob = F.log_softmax(out, dim=1) #[(batch_size*beam_size) x vocab_size]           
            score = score.view(-1, 1) + log_prob# [(batch_size*beam_size)] => [(batch_size*beam_size) x vocab_size]
            
            # Select `beam size` transitions out of `vocab size` combinations
            # [(batch_size*beam_size) x vocab_size]=> [batch_size x (beam_size*vocab_size)]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_size]
            # top_k_idx: [batch_size, beam_size], each element of top_k_idx [0 ~ beam x vocab)
            score, top_k_idx = score.view(batch_size, -1).topk(beam_size, dim=1)

            # Get token ids with remainder after dividing by top_k_idx. Each element is among [0, vocab_size) 
            # Ex. Index of token 3 in beam 4 : (4 * vocab size) + 3 => 3
            x = (top_k_idx % self.vocab_size).view(-1) # x: [(batch_size*beam_size)]

            # top-k-pointer [(batch_size*beam_size)]
            #       Points top-k beam that scored best at current step
            #       Later used as back-pointer at backtracking
            #       Each element is beam index: 0 ~ beam_size + position index: 0 ~ beam_size*(batch_size-1)
            beam_idx = top_k_idx / self.vocab_size  # [batch_size, beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # Select next h (size doesn't change)            
            h = h.index_select(1, top_k_pointer)# [num_layers x (batch_size*beam_size) x hidden_size]
            
            beam.update(score.clone(), top_k_pointer, x)  # , h)# Update sequence scores at beam

            # Erase scores for EOS so that they are not expanded
            eos_idx = x.data.eq(EOS_ID).view(batch_size, beam_size) # [batch_size x beam_size]
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack() # prediction: [batch_size x n_samples x seq_len]
        prediction = prediction.data.cpu().numpy()
        final_score = final_score.data.cpu().numpy()
        
        return prediction, length, final_score
    
    
class Beam(object):
    def __init__(self, batch_size, beam_size, max_unroll, batch_position):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_unroll = max_unroll

        # batch_position [batch_size]
        #   [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]
        #   Points where batch starts in [batch_size x beam_size] tensors
        #   Ex. position_idx[5]: when 5-th batch starts
        self.batch_position = batch_position

        self.log_probs = list()  # [(batch*k, vocab_size)] * sequence_length
        self.scores = list()  # [(batch*k)] * sequence_length
        self.back_pointers = list()  # [(batch*k)] * sequence_length
        self.token_ids = list()  # [(batch*k)] * sequence_length
        # self.hidden = list()  # [(num_layers, batch*k, hidden_size)] * sequence_length


    def update(self, score, back_pointer, token_id):  # , h):
        """Append intermediate top-k candidates to beam at each step"""
        self.scores.append(score)
        self.back_pointers.append(back_pointer)
        self.token_ids.append(token_id)

    def backtrack(self):
        """Backtracks over batch to generate optimal k-sequences
        Returns:
            prediction ([batch, k, max_unroll]) - A list of Tensors containing predicted sequence
            final_score [batch, k] - A list containing the final scores for all top-k sequences
            length [batch, k] - A list specifying the length of each sequence in the top-k candidates
        """
        prediction = list()
        
        length = [[self.max_unroll] * self.beam_size for _ in range(self.batch_size)]# Initialize for length of top-k sequences

        # Last step output of the beam are not sorted => sort here!
        # Size not changed [batch size, beam_size]
        top_k_score, top_k_idx = self.scores[-1].topk(self.beam_size, dim=1)

        top_k_score = top_k_score.clone()# Initialize sequence scores

        n_eos_in_batch = [0] * self.batch_size

        # Initialize Back-pointer from the last step
        # Add self.position_idx for indexing variable with batch x beam as the first dimension
        back_pointer = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)# [batch*beam]

        for t in reversed(range(self.max_unroll)):
                       
            token_id = self.token_ids[t].index_select(0, back_pointer) # [batch*beam] # Reorder variables with the Back-pointer
            back_pointer = self.back_pointers[t].index_select(0, back_pointer)# [batch*beam]  # Reorder the Back-pointer                   
            eos_indices = self.token_ids[t].data.eq(EOS_ID).nonzero()# [< batch*beam] # Indices of ended sequences 

            # For each batch, every time we see an EOS in the backtracking process,
            # If not all sequences are ended
            #    lowest scored survived sequence <- detected ended sequence
            # if all sequences are ended
            #    lowest scored ended sequence <- detected ended sequence
            if eos_indices.dim() > 0:                
                for i in range(eos_indices.size(0) - 1, -1, -1):  # Loop over all EOS at current step                  
                    eos_idx = eos_indices[i, 0].item()# absolute index of detected ended sequence

                    # At which batch EOS is located
                    batch_idx = eos_idx // self.beam_size
                    batch_start_idx = batch_idx * self.beam_size

                    # if n_eos_in_batch[batch_idx] > self.beam_size:

                    # Index of sequence with lowest score
                    _n_eos_in_batch = n_eos_in_batch[batch_idx] % self.beam_size
                    beam_idx_to_be_replaced = self.beam_size - _n_eos_in_batch - 1
                    idx_to_be_replaced = batch_start_idx + beam_idx_to_be_replaced

                    # Replace old information with new sequence information
                    back_pointer[idx_to_be_replaced] = self.back_pointers[t][eos_idx].item()
                    token_id[idx_to_be_replaced] = self.token_ids[t][eos_idx].item()
                    top_k_score[batch_idx,
                                beam_idx_to_be_replaced] = self.scores[t].view(-1)[eos_idx].item()
                    length[batch_idx][beam_idx_to_be_replaced] = t + 1

                    n_eos_in_batch[batch_idx] += 1
            
            prediction.append(token_id)# max_unroll * [batch x beam]

        # Sort and re-order again as the added ended sequences may change the order        
        top_k_score, top_k_idx = top_k_score.topk(self.beam_size, dim=1)# [batch, beam]
        final_score = top_k_score.data

        for batch_idx in range(self.batch_size):
            length[batch_idx] = [length[batch_idx][beam_idx.item()]
                                 for beam_idx in top_k_idx[batch_idx]]
        
        top_k_idx = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)# [batch x beam]

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in the reverse order        
        prediction = [step.index_select(0, top_k_idx).view(
            self.batch_size, self.beam_size) for step in reversed(prediction)]# [batch, beam]

        prediction = torch.stack(prediction, 2)# [batch, beam, max_unroll]

        return prediction, final_score, length
    
