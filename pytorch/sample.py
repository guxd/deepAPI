import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import indexes2sent, gVar, gData
import model, data, configs
from metrics import Metrics
from data_loader import APIDataset, load_dict, load_vecs


def evaluate(model, metrics, test_loader, vocab_desc, vocab_api, f_eval, repeat):
    ivocab_api = {v: k for k, v in vocab_api.items()}
    ivocab_desc = {v: k for k, v in vocab_desc.items()}
    
    recall_bleus, prec_bleus = [], []
    local_t = 0
    for descs, apiseqs, desc_lens, api_lens in tqdm(test_loader):
        
        if local_t>2000:
            break
        
        
        desc_str = indexes2sent(descs[0].numpy(), vocab_desc)
        
        descs, desc_lens = gVar(descs), gVar(desc_lens)
        sample_words, sample_lens = model.sample(descs, desc_lens, repeat)
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words, vocab_api)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        ref_str, _ =indexes2sent(apiseqs[0].numpy(), vocab_api, vocab_api["<s>"])
        ref_tokens = ref_str.split(' ')
        
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)
        
        local_t += 1 
        f_eval.write("Batch %d \n" % (local_t))# print the context        
        f_eval.write("Query: {} \n".format(desc_str))
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))# print the true outputs 
        for r_id, pred_sent in enumerate(pred_sents):
            f_eval.write("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2*(prec_bleu*recall_bleu) / (prec_bleu+recall_bleu+10e-12)
    
    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f"% (recall_bleu, prec_bleu, f1)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")
    
    return recall_bleu, prec_bleu

def main(args):
    conf = getattr(configs, 'config_'+args.model)()
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")
    
    # Load data
    test_set=APIDataset(args.data_path+'test.desc.h5', args.data_path+'test.apiseq.h5', conf['maxlen'])
    test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    vocab_api = load_dict(args.data_path+'vocab.apiseq.json')
    vocab_desc = load_dict(args.data_path+'vocab.desc.json')
    n_tokens = len(vocab_api)

    metrics=Metrics()
    
    # Load model checkpoints   
    import model
    model = getattr(model, args.model)(conf, n_tokens)
    ckpt='./output/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.reload_from)
    model.load_state_dict(torch.load(ckpt))
    if torch.cuda.is_available():
        model=model.cuda()
    model.eval()
    
    f_eval = open("./output/{}/{}/results.txt".format(args.model, args.expname), "w")
    repeat = args.n_samples
    
    evaluate(model, metrics, test_loader, vocab_desc, vocab_api, f_eval, repeat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DeepAPI for Eval')
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='DeepAPI', help='model name')
    parser.add_argument('--expname', type=str, default='basic', help='experiment name, disinguishing different parameter settings')
    parser.add_argument('--reload_from', type=int, default=1000, help='directory to load models from')
    
    parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    print(vars(args))
    main(args)
