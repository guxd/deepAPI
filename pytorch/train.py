import argparse
import time
from datetime import datetime
import numpy as np
import random
import json
import logging
import torch
import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import timeSince, sent2indexes, indexes2sent, gData, gVar
import model, configs, data_loader
from data_loader import APIDataset, APIDataset, load_dict, load_vecs
from metrics import Metrics
from sample import evaluate

from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

parser = argparse.ArgumentParser(description='DeepAPI Pytorch')
# Path Arguments
parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='DeepAPI', help='model name')
parser.add_argument('--expname', type=str, default='basic', help='experiment name, for disinguishing different parameter settings')
parser.add_argument('--visual', action='store_true', default=False, help='visualize training status in tensorboard')
parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
parser.add_argument('--log_every', type=int, default=10, help='interval to log autoencoder training results')
parser.add_argument('--valid_every', type=int, default=100, help='interval to validation')
parser.add_argument('--eval_every', type=int, default=2000, help='interval to evaluation to concrete results')
parser.add_argument('--seed', type=int, default=1111, help='random seed')

args = parser.parse_args()
print(vars(args))

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.model)):
    os.makedirs('./output/{}'.format(args.model))
if not os.path.isdir('./output/{}/{}'.format(args.model, args.expname)):
    os.makedirs('./output/{}/{}'.format(args.model, args.expname))
if not os.path.isdir('./output/{}/{}/models'.format(args.model, args.expname)):
    os.makedirs('./output/{}/{}/models'.format(args.model, args.expname))
if not os.path.isdir('./output/{}/{}/tmp_results'.format(args.model, args.expname)):
    os.makedirs('./output/{}/{}/tmp_results'.format(args.model, args.expname))

# save arguments
json.dump(vars(args), open('./output/{}/{}/args.json'.format(args.model, args.expname), 'w'))

# LOG #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")#,format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
fh = logging.FileHandler("./output/{}/{}/logs.txt".format(args.model, args.expname))
                                  # create file handler which logs even debug messages
logger.addHandler(fh)# add the handlers to the logger

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu_id) # set gpu device
    torch.cuda.manual_seed(args.seed)

def save_model(model, epoch):
    """Save model parameters to checkpoint"""
    ckpt_path='./output/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, epoch)
    print(f'Saving model parameters to {ckpt_path}')
    torch.save(model.state_dict(), ckpt_path)

def load_model(model, epoch):
    """Load parameters from checkpoint"""
    ckpt_path='./output/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, epoch)
    print(f'Loading model parameters from {ckpt_path}')
    model.load_state_dict(torch.load(checkpoint))

config = getattr(configs, 'config_'+args.model)()

###############################################################################
# Load data
###############################################################################
train_set=APIDataset(args.data_path+'train.desc.h5', args.data_path+'train.apiseq.h5', config['maxlen'])
valid_set=APIDataset(args.data_path+'test.desc.h5', args.data_path+'test.apiseq.h5', config['maxlen'])

vocab_api = load_dict(args.data_path+'vocab.apiseq.json')
vocab_desc = load_dict(args.data_path+'vocab.desc.json')
n_tokens = len(vocab_api)

metrics=Metrics()

print("Loaded data!")

###############################################################################
# Define the models
###############################################################################

model = getattr(model, args.model)(config, n_tokens) 
if args.reload_from>=0:
    load_model(model, args.reload_from)
    
if use_cuda:
    model=model.cuda()

tb_writer = SummaryWriter("./output/{}/{}/logs/".format(args.model, args.expname)\
                          +datetime.now().strftime('%Y%m%d%H%M')) if args.visual else None

logger.info("Training...")
itr_global=1
start_epoch=1 if args.reload_from==-1 else args.reload_from+1
for epoch in range(start_epoch, config['epochs']+1):

    epoch_start_time = time.time()
    itr_start_time = time.time()
    
    # shuffle (re-define) data between epochs   
    train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    train_data_iter=iter(train_loader)
    n_iters=train_data_iter.__len__()
    
    itr = 1
    while True:# loop through all batches in training data
        model.train()
        try:
            descs, apiseqs, desc_lens, api_lens = train_data_iter.next() 
        except StopIteration: # end of epoch
            break 
        descs, apiseqs, desc_lens, api_lens = gVar(descs), gVar(apiseqs), gVar(desc_lens), gVar(api_lens)
        loss_AE = model.train_AE(descs, desc_lens, apiseqs, api_lens)                  
                               
        if itr % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            log = '%s-%s|@gpu%d epo:[%d/%d] iter:[%d/%d] step_time:%ds elapsed:%s \n                      '\
            %(args.model, args.expname, args.gpu_id, epoch, config['epochs'],
                     itr, n_iters, elapsed, timeSince(epoch_start_time,itr/n_iters))
            for loss_name, loss_value in loss_AE.items():
                log=log+loss_name+':%.4f '%(loss_value)
                if args.visual:
                    tb_writer.add_scalar(loss_name, loss_value, itr_global)
            logger.info(log)
                
            itr_start_time = time.time()   
            
        if itr % args.valid_every == 0:
            valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
            model.eval()
            loss_records={}
            
            for descs, apiseqs, desc_lens, api_lens in valid_loader:
                descs, apiseqs, desc_lens, api_lens = gVar(descs), gVar(apiseqs), gVar(desc_lens), gVar(api_lens)
                valid_loss = model.valid(descs, desc_lens, apiseqs, api_lens)    
                for loss_name, loss_value in valid_loss.items():
                    v=loss_records.get(loss_name, [])
                    v.append(loss_value)
                    loss_records[loss_name]=v
                
            log = 'Validation '
            for loss_name, loss_values in loss_records.items():
                log = log + loss_name + ':%.4f  '%(np.mean(loss_values))
                if args.visual:
                    tb_writer.add_scalar(loss_name, np.mean(loss_values), itr_global)                 
            logger.info(log)    
            
        itr += 1
        itr_global+=1
        
        
        if itr_global % args.eval_every == 0:  # evaluate the model in the develop set
            model.eval()               
            valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)
        
            f_eval = open("./output/{}/{}/tmp_results/iter{}.txt".format(args.model, args.expname, itr_global), "w")
            repeat = 10
        
            recall_bleu, prec_bleu = evaluate(model, metrics, valid_loader, vocab_desc, vocab_api, f_eval, repeat)
                         
            if args.visual:
                tb_writer.add_scalar('recall_bleu', recall_bleu, itr_global)
                tb_writer.add_scalar('prec_bleu', prec_bleu, itr_global)
            
            save_model(model, itr_global) # save model after each epoch
        
    # end of epoch ----------------------------
    model.adjust_lr()
    

    

