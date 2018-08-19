'''Original code: https://github.com/tocab/NaturalResponseGeneration/blob/master/metrics/metrics.py'''
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import Counter

def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)])
        yield sum((s_ngrams & r_ngrams).values())
        yield max(len(hypothesis) + 1 - n, 0)

def smoothed_bleu(stats):
    small = 1e-9
    tiny = 1e-15 ## so that if guess is 0 still return 0
    c, r = stats[:2]
    log_bleu_prec = sum([np.log((tiny + float(x)) / (small + y)) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return np.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100

class Metrics:
    """
    """
    def __init__(self):
        """
        :param word2vec - a numpy array of word2vec with shape [vocab_size x emb_size]
        """
        super(Metrics, self).__init__()

    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
    
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
               # scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
               #                         weights=[1./4, 1./4, 1./4, 1./4]))
                scores.append(smoothed_bleu(list(bleu_stats(hyp, ref))))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)
    




