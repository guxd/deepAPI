#!/usr/bin/env python
#sample translations from the model (or to find the most probable translations)
import argparse
import pickle
import json
import traceback
import logging
import time
import sys
import matplotlib.pyplot as plt

import numpy

from model import RNNEncoderDecoder
from state import prototype_state
from helper import parse_input

from numpy import argpartition
from analysis import bleu_analyze

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']
        
#         self.wordidf_dict_src = json.loads(open(state['word_weight'], "r").readline())         
#         #plt.hist(self.wordidf_dict_src.values())
#         #plt.show()
        self.wordidf_dict_tar = json.loads(open(state['word_weight_trgt'],'r').readline())
#         max_wordidx=max(self.wordidf_dict_src.keys())
#         self.wordidf_array_src=numpy.zeros(max_wordidx+2)
#         for idx in self.wordidf_dict_src.keys():
#             if idx!=None:
#                 self.wordidf_array_src[idx]=self.wordidf_dict_src.get(idx,0.0)   
        

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
        c = self.comp_repr(seq)[0] #compute the context vector c of a sequence
        states = list(map(lambda x : x[None, :], self.comp_init_states(c)))
        dim = states[0].shape[1]
        
#         meanidf_src=self.wordidf_array_src[seq[0:len(seq)-1]].mean()
#         print meanidf_src
#         print self.wordidf_array_src[seq[0:len(seq)-1]]
        
        num_levels = len(states)#number of hidden layers

        final_trans = []#final translations
        final_costs = []#corresponding costs

        trans = [[]]# current temp translations (partial sequences) in the beam
        costs = [0.0]# corresponding cost for each temp translation

        for k in range(12 * len(seq)):#maximum number of expansion is 12 times of source sequence length.
            if n_samples == 0: break
            # Compute probabilities of the next words for all the elements of the beam.
            beam_size = len(trans)# real beam size, beam size is reduced when a satisfied result is found, so it is updated at each step
            last_words = (numpy.array(list(map(lambda t : t[-1], trans)))#get the last word for each temp translation sequence
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states)[0])
            #compute the prob of all next words given the context vector c, the time step k, current hidden states, and each 'last word'
            #the resulting log probs is s*v dimension, where s is the number of current temp translations(beam size) and v is vocabulary size
            
            if k>0:
                for wordid in self.wordidf_dict_tar.keys():
                    log_probs[:,wordid]+=0.035*self.wordidf_dict_tar.get(wordid,0.0)
                    
            
            # Adjust log probs according to search restrictions
            if ignore_unk: log_probs[:,self.unk_id] = -numpy.inf
            if k < minlen: log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(flat_next_costs.flatten(),n_samples)[:n_samples]
                #indices of the temp mig sequences with the best costs
            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices // voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]#append 'next_word' to current temp translation
                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:#found a satisfied translation, add it to final retults
                    n_samples -= 1
                    final_trans.append(new_trans[i])
                    final_costs.append(new_costs[i])
            states = list(map(lambda x : x[indices], new_states))#update states

        # Dirty tricks to obtain any translation
        if not len(final_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else: logger.error("Translation failed")

        final_trans = numpy.array(final_trans)[numpy.argsort(final_costs)]
        final_costs = numpy.array(sorted(final_costs))
        return c, final_trans, final_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '</s>': break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples, sampler=None, beam_search=None,
        ignore_unk=False, normalize=False, alpha=1, verbose=False):
    if beam_search:
        sentences = []
        c, trans, costs = beam_search.search(seq, n_samples, ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose: print ("{}: {}".format(costs[i], sentences[i]))
        return c, sentences, costs, trans
    elif sampler:
        sentences = []
        all_probs = []
        costs = []

        c,values, cond_probs = sampler(n_samples, 3 * (len(seq) - 1), alpha, seq)
        for sidx in xrange(n_samples):
            sen = []
            for k in xrange(values.shape[0]):
                if lm_model.word_indxs[values[k, sidx]] == '</s>': break
                sen.append(lm_model.word_indxs[values[k, sidx]])
            sentences.append(" ".join(sen))
            probs = cond_probs[:, sidx]
            probs = numpy.array(cond_probs[:len(sen) + 1, sidx])
            all_probs.append(numpy.exp(-probs))
            costs.append(-numpy.sum(probs))
        if normalize:
            counts = [len(s.strip().split(" ")) for s in sentences]
            costs = [co / cn for co, cn in zip(costs, counts)]
        sprobs = numpy.argsort(costs)
        if verbose:
            for pidx in sprobs:
                print ("{}: {} {} {}".format(pidx, -costs[pidx], all_probs[pidx], sentences[pidx]))
            print()
        return c, sentences, costs, None
    else:
        raise Exception("I don't know what to do")


def parse_args():
    '''--state ./data/search_desc2apiseq_state.pkl 
    --model_path ./data/search_desc2apiseq_model.npz 
    --source ./data/test.desc.txt 
    --trans ./data//result.apiseq.txt 
    --validate ./data/test.apiseq.txt'''
    defaultfolder='./data/'
    defaultstate=defaultfolder+'search_desc2apiseq_state.pkl'
    defaultmodel=defaultfolder+'search_desc2apiseq_model.npz'
    defaultsource=defaultfolder+'test.desc.txt'
    defaultresult=defaultfolder+'result.apiseq.txt'
    defaultvalid=defaultfolder+'test.apiseq.txt'
    defaultvecfile=defaultfolder+'result.vec.txt'
    
    parser = argparse.ArgumentParser("Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state", required=False, help="State to use", default=defaultstate)
    parser.add_argument("--beam-search", action="store_true", default=True, help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",  type=int, help="Beam size", default=12)
    parser.add_argument("--ignore-unk", default=False, action="store_true", help="Ignore unknown words")
    parser.add_argument("--source", default=defaultsource, help="File of source sentences")
    parser.add_argument("--trans", default=defaultresult, help="File to save translations in")
    parser.add_argument("--top_num", type=int, help="the number of top results for each source", default=1)
    parser.add_argument("--validate",default=defaultvalid, help="File of validation references")
    parser.add_argument("--vec",default=defaultvecfile,help="File of resulting context vectors")
    parser.add_argument("--normalize",action="store_true", default=False, help="Normalize log-prob with the word count")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    parser.add_argument("--model_path", default=defaultmodel, help="Path to the model")
    parser.add_argument("changes", nargs="?", default="", help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()
    # Sample args: 
    # --state .\data\phrase_state.pkl  
    # --beam-search --model_path .\data\github\phrase_model.npz

    state = prototype_state()
    with open(args.state, 'rb') as src:
        state.update(pickle.load(src))
    state.update(eval("dict({})".format(args.changes)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = json.loads(open(state['word_indx'],'r').readline())
    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec)
        beam_search.compile()
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    idict_src = {v: k for k, v in indx_word.items()}

    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w') 
        if args.vec: fvec=open(args.vec,'w')
        top_num=args.top_num       
            
        start_time = time.time()

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            if args.verbose: print ("Parsed Input {}:".format(i), parsed_in)
            context_vec, trans, costs, _ = sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
            if not trans:#if no translation
                for ss in range(top_num):
                    print >>ftrans, "a"
            else:                
                top = numpy.array(costs).argsort()[0:top_num]  
                total_cost += costs[top[0]]                          
                for k in top: print >>ftrans, trans[k]
                if len(top)<top_num:
                    for ss in range(top_num-len(top)): print >>ftrans, "a"
            if args.verbose and trans:
                print ("Translation:{}".format(trans[top[0]]))
                #print ("Context Vector:%d",context_vec)
                
            if args.vec:#print context vectors
                numpy.set_printoptions(threshold='nan',suppress=True,precision=12, linewidth=100000)
                if state['forward']:
                    assert context_vec.shape[1] >= state['dim']
                    forwardvec=context_vec[-1][0:state['dim']]
                    vec=forwardvec
                    if state['backward']:
                        assert context_vec.shape[1] == 2*state['dim']
                        backwardvec=context_vec[0][state['dim']:2*state['dim']]
                        vec=numpy.concatenate((forwardvec,backwardvec))
                    print >>fvec,vec
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                if args.vec: fvec.flush()
                logger.debug("Current speed is {} per sentence".format((time.time() - start_time) / (i + 1)))
        print ("Total cost of the translations: {}".format(total_cost))
        fsrc.close()
        ftrans.close()
        if args.vec: fvec.close()
        
        '''Validate the results and show BLEU results'''     
        if args.validate:       
            ftrans = open(args.trans, 'r')
            fvalid=open(args.validate, 'r')            
            avg_bleu=bleu_analyze(ftrans.readlines(),fvalid.readlines(),top_num)
            ftrans.close()
            fvalid.close()
            print ("Avg bleu of the translations: {}".format(avg_bleu))      
        
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print ("Parsed Input: {}".format(parsed_in))
            except Exception:
                print ("Exception while parsing your input:")
                traceback.print_exc()
                continue

            sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search,
                    ignore_unk=args.ignore_unk, normalize=args.normalize,
                    alpha=alpha, verbose=True)

if __name__ == "__main__":
    main()
