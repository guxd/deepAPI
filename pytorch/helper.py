######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math
import numpy as np

def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m /60)
    m -= h *60
    return '%d:%d:%d'% (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asHHMMSS(s), asHHMMSS(rs))


PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]


#######################################################################
import nltk
try: nltk.word_tokenize("hello world")
except LookupError: nltk.download('punkt')
    
def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''      
    def convert_sent(sent, vocab, maxlen):
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        tokens = nltk.word_tokenize(sent.strip())
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len): idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len
    if type(sentence) is list:
        inds, lens = None, None
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            idxes, idx_len = np.expand_dims(idxes, 0), np.array([idx_len])
            inds = idxes if inds is None else np.concatenate((inds, idxes))
            lens = idx_len if lens is None else np.concatenate((lens, idx_len))
        return inds, lens
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]

def indexes2sent(indexes, vocab, ignore_tok=PAD_ID): 
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, ignore_tok=PAD_ID):
        toks=[]
        length=0
        indexes=filter(lambda i: i!=ignore_tok, indexes)
        for idx in indexes:
            toks.append(ivocab[idx])
            length+=1
            if idx == EOS_ID: break
        return ' '.join(toks), length
    
    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim==1:# one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:# dim>1
        sentences=[] # a batch of sentences
        lens=[]
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens

    
#########################################################################
########  For Sequence Padding and Masking #######
import torch

def get_position(lens, maxlen, left_pad=False):
    """ transform sequence length to a series of positions. e.g., 3 -> 1,2,3"""
    batch_size = lens.size(0)
    pos = torch.zeros((batch_size, maxlen), dtype=torch.long, device=lens.device)
    for i, input_len in enumerate(lens):
        if not left_pad:
            pos[i,:input_len] = torch.arange(1, input_len+1, dtype=torch.long, device=lens.device)
        else:
            pos[i,maxlen-input_len:] = torch.arange(1, input_len+1, dtype=torch.long, device=lens.device)
    return pos

def sequence_mask(sequence_length, max_len=None):
    '''
    Convert sequence lengths to masking vectors
    '''
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=sequence_length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand
