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
from queue import PriorityQueue
import operator
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
        score_unnormalized = self.score_linear(torch.tanh(score_before_transformation))
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
        self.dropout= dropout
        
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
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(self.out.weight)        
        nn.init.constant_(self.out.bias, 0.)
    
    def forward(self, init_h, enc_hids, context=None, inputs=None, lens=None):
        '''
        init_h: initial hidden state for decoder
        enc_hids: a tuple of (enc_hids, mask) for attention use
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
            enc_hids, enc_hids_mask = enc_hids
            _, attn_ctx = self.attn(enc_hids, [h.transpose(0,1)], enc_hids_mask)# initial attention context
            for di in range(maxlen):
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
        return decoded, h
    
    def sampling(self, init_h, enc_hids, context, maxlen, mode='greedy', to_numpy=True):
        """
        A simple greedy sampling
        :param init_h: [batch_sz x hid_sz]
        :param enc_hids: a tuple of (enc_hids, mask) for attention use. [batch_sz x seq_len x hid_sz]
        """
        batch_size=init_h.size(0)
        decoded_words = gVar(torch.zeros(batch_size, maxlen)).long()  
        sample_lens, len_inc = gVar(torch.zeros(batch_size)).long(), gVar(torch.ones(batch_size)).long()
               
        x = gVar(torch.LongTensor([[SOS_ID]*batch_size]).view(batch_size,1))# [batch_sz x 1] (1=seq_len)
        h = init_h.unsqueeze(0) # [1 x batch_sz x hid_sz]   
        for di in range(maxlen):               
            out, h = self.forward(h.squeeze(0), enc_hids, context, x)
            if mode=='greedy':
                x = out[:,-1].max(1, keepdim=True)[1] # x:[batch_sz x 1] indexes of predicted words
            elif mode=='sample':
                x = torch.multinomial(F.softmax(out[:,-1], dim=1), 1)    
            decoded_words[:,di] = x.squeeze()
            len_inc=len_inc*(x.squeeze()!=EOS_ID).long() # stop increse length (set 0 bit) when EOS is met
            sample_lens=sample_lens+len_inc            
        
        if to_numpy:
            decoded_words = decoded_words.data.cpu().numpy()
            sample_lens = sample_lens.data.cpu().numpy()
        return decoded_words, sample_lens
    
    def beam_decode(self, init_h, enc_hids, context, beam_width, max_unroll, topk=1):
        '''
        https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        :param init_h: input tensor of shape [B, H] for start of the decoding
        :param enc_hids: if you are using attention mechanism you can pass encoder outputs, [B, T, H] where T is the maximum length of input sentence
        :param topk: how many sentence do you want to generate
        :return: decoded_batch
        '''        
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
                enc_outs, enc_outs_mask = enc_hids
                enc_outs = enc_outs[idx,:,:].unsqueeze(0) 
                enc_outs_mask = enc_outs_mask[idx, :].unsqueeze(0)
                enc_outs = (enc_outs, enc_outs_mask)

            # Start with the start of the sentence token
            x = gVar(torch.LongTensor([[SOS_ID]]))

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk-len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(h, None, x, 0, 1)
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
                out, h = self.forward(h.squeeze(0), enc_outs, None, x) # out [1 x 1 x vocab_size]
                out = out.squeeze(1)# [1 x vocab_size]
                out = F.log_softmax(out, 1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(out, beam_width) # [1 x beam_width]

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()
                    node = BeamSearchNode(h, n, decoded_t, n.logp + log_p, n.len + 1)
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


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, other):
        '''overwrith lt function. when two nodes have the same score, the priority queue will compare the two nodes themselves.'''
        print("[warning] two nodes (%d-%d) have the same score, possible recursive (cycle) search."%(self.wordid, other.wordid) )
        return True