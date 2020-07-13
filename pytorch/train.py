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
from helper import timeSince, sent2indexes, indexes2sent
import models, configs, data_loader
from data_loader import APIDataset, APIDataset, load_dict, load_vecs
from metrics import Metrics
from sample import evaluate
from modules import get_cosine_schedule_with_warmup

from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package


def train(args):
    timestamp=datetime.now().strftime('%Y%m%d%H%M')    
    # LOG #
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")#,format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    tb_writer=None
    if args.visual:
        # make output directory if it doesn't already exist
        os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/models', exist_ok=True)
        os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/temp_results', exist_ok=True)
        fh = logging.FileHandler(f"./output/{args.model}/{args.expname}/{timestamp}/logs.txt")
                                      # create file handler which logs even debug messages
        logger.addHandler(fh)# add the handlers to the logger
        tb_writer = SummaryWriter(f"./output/{args.model}/{args.expname}/{timestamp}/logs/")
        # save arguments
        json.dump(vars(args), open(f'./output/{args.model}/{args.expname}/{timestamp}/args.json', 'w'))

    # Device #
    if args.gpu_id<0: 
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id>-1 else "cpu")
    print(device)
    n_gpu = torch.cuda.device_count() if args.gpu_id<0 else 1
    print(f"num of gpus:{n_gpu}")
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def save_model(model, epoch, timestamp):
        """Save model parameters to checkpoint"""
        os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/models', exist_ok=True)
        ckpt_path=f'./output/{args.model}/{args.expname}/{timestamp}/models/model_epo{epoch}.pkl'
        print(f'Saving model parameters to {ckpt_path}')
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, epoch, timestamp):
        """Load parameters from checkpoint"""
        ckpt_path=f'./output/{args.model}/{args.expname}/{timestamp}/models/model_epo{epoch}.pkl'
        print(f'Loading model parameters from {ckpt_path}')
        model.load_state_dict(torch.load(checkpoint))

    config = getattr(configs, 'config_'+args.model)()

    ###############################################################################
    # Load data
    ###############################################################################
    train_set=APIDataset(args.data_path+'train.desc.h5', args.data_path+'train.apiseq.h5', config['max_sent_len'])
    valid_set=APIDataset(args.data_path+'test.desc.h5', args.data_path+'test.apiseq.h5', config['max_sent_len'])
    train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    print("Loaded data!")

    ###############################################################################
    # Define the models
    ###############################################################################
    model = getattr(models, args.model)(config) 
    if args.reload_from>=0:
        load_model(model, args.reload_from)
    model=model.to(device)
    
    
    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=config['adam_epsilon'])        
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config['warmup_steps'], 
            num_training_steps=len(train_loader)*config['epochs']) # do not foget to modify the number when dataset is changed

    ###############################################################################
    # Training
    ###############################################################################
    logger.info("Training...")
    itr_global=1
    start_epoch=1 if args.reload_from==-1 else args.reload_from+1
    for epoch in range(start_epoch, config['epochs']+1):

        epoch_start_time = time.time()
        itr_start_time = time.time()

        # shuffle (re-define) data between epochs   

        for batch in train_loader:# loop through all batches in training data
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)  
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                log = '%s-%s|@gpu%d epo:[%d/%d] iter:%d step_time:%ds loss:%f'\
                %(args.model, args.expname, args.gpu_id, epoch, config['epochs'],itr_global, elapsed, loss)
                if args.visual:
                        tb_writer.add_scalar('loss', loss, itr_global)
                logger.info(log)

                itr_start_time = time.time()   

            if itr_global % args.valid_every == 0:
             
                model.eval()
                loss_records={}

                for batch in valid_loader:
                    batch_gpu = [tensor.to(device) for tensor in batch]
                    with torch.no_grad():
                        valid_loss = model.valid(*batch_gpu)    
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

            itr_global+=1        

            if itr_global % args.eval_every == 0:  # evaluate the model in the develop set
                model.eval()      
                save_model(model, itr_global, timestamp) # save model after each epoch
                
                valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)
                vocab_api = load_dict(args.data_path+'vocab.apiseq.json')
                vocab_desc = load_dict(args.data_path+'vocab.desc.json')
                metrics=Metrics()
                
                os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/temp_results', exist_ok=True)
                f_eval = open(f"./output/{args.model}/{args.expname}/{timestamp}/temp_results/iter{itr_global}.txt", "w")
                repeat = 1
                decode_mode = 'sample'
                recall_bleu, prec_bleu = evaluate(model, metrics, valid_loader, vocab_desc, vocab_api, repeat, decode_mode, f_eval)

                if args.visual:
                    tb_writer.add_scalar('recall_bleu', recall_bleu, itr_global)
                    tb_writer.add_scalar('prec_bleu', prec_bleu, itr_global)
                

        # end of epoch ----------------------------
        model.adjust_lr()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DeepAPI Pytorch')
    # Path Arguments
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='RNNEncDec', help='model name: RNNEncDec')
    parser.add_argument('--expname', type=str, default='basic', help='experiment name, for disinguishing different parameter settings')
    parser.add_argument('-v', '--visual', action='store_true', default=False, help='visualize training status in tensorboard')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    # Evaluation Arguments
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--log_every', type=int, default=10, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=1000, help='interval to validation')
    parser.add_argument('--eval_every', type=int, default=5000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn
    
    train(args)

