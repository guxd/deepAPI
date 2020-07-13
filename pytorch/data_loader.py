import sys
import json
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import pickle
import random
import numpy as np
from helper import PAD_ID, SOS_ID, EOS_ID, UNK_ID

use_cuda = torch.cuda.is_available()

    
class APIDataset(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, desc_file, api_file, max_seq_len=50):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.max_seq_len=max_seq_len
        
        print("loading data...")
        api_table = tables.open_file(api_file)
        self.api_data = api_table.get_node('/phrases')[:].astype(np.long)
        self.api_index = api_table.get_node('/indices')[:]
        
        desc_table = tables.open_file(desc_file)
        self.desc_data = desc_table.get_node('/phrases')[:].astype(np.long)
        self.desc_index = desc_table.get_node('/indices')[:]
        
        assert self.api_index.shape[0] == self.desc_index.shape[0], "inconsistent number of API sequences and NL descriptions!"
        self.data_len = self.api_index.shape[0]
        print("{} entries".format(self.data_len))
        
    def list2array(self, L, max_len, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        arr = np.zeros(max_len, dtype=dtype)+pad_idx
        for i, v in enumerate(L): arr[i] = v
        return arr
    
    def __getitem__(self, offset):
        pos, api_len = self.api_index[offset]['pos'], self.api_index[offset]['length']
        api_len = min(int(api_len),self.max_seq_len-2) # real length of sequences
        api = [SOS_ID] + self.api_data[pos: pos + api_len].tolist() + [EOS_ID]
        
        pos, desc_len = self.desc_index[offset]['pos'], self.desc_index[offset]['length']
        desc_len=min(int(desc_len),self.max_seq_len-2) # get real seq len
        desc= [SOS_ID] + self.desc_data[pos: pos+ desc_len].tolist() + [EOS_ID]
       
        ## Padding ##
        api = self.list2array(api, self.max_seq_len, np.int, PAD_ID)
        desc= self.list2array(desc, self.max_seq_len, np.int, PAD_ID)
        
        return desc, desc_len, api, api_len

    def __len__(self):
        return self.data_len
    
    
def load_dict(filename):
    return json.loads(open(filename, "r", encoding="utf-8").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    
    input_dir='./data/'
    VALID_FILE_API=input_dir+'test.apiseq.shuf.h5'
    VALID_FILE_DESC=input_dir+'test.desc.shuf.h5'
    valid_set=APIDataset(VALID_FILE_DESC, VALID_FILE_API)
    valid_data_loader=torch.utils.data.DataLoader(dataset=valid_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)
    vocab_api = load_dict(input_dir+'vocab.apiseq.pkl')
    ivocab_api = {v: k for k, v in vocab_api.items()}
    vocab_desc = load_dict(input_dir+'vocab.desc.pkl')
    ivocab_desc = {v: k for k, v in vocab_desc.items()}
    #print ivocab
    k=0
    for qapair in valid_data_loader:
        k+=1
        if k>20:
            break
        decoded_words=[]
        idx=qapair[0].numpy().tolist()[0]
        print (idx)
        for i in idx:
            decoded_words.append(ivocab_desc[i])
        question = ' '.join(decoded_words)
        decoded_words=[]
        idx=qapair[1].numpy().tolist()[0]
        for i in idx:
            decoded_words.append(ivocab_api[i])
        answer=' '.join(decoded_words)
        print('<', question)
        print('>', answer)
