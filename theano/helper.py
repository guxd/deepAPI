import numpy
import logging
logger = logging.getLogger(__name__)

import operator
import itertools
from groundhog.datasets import PytablesBitextIterator
def create_padded_batch(state, x, y, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of numpy.array's, each array is a batch of phrases in some of source languages

    :type y: list
    :param y: same as x but for target languages

    :param new_format: a wrapper to be applied on top of returned value

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple

    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    """

    mx = state['seqlen']
    my = state['seqlen']
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = numpy.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = numpy.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = numpy.zeros((mx, n), dtype='int64')
    Y = numpy.zeros((my, n), dtype='int64')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    Ymask = numpy.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in range(len(x[0])):
        # Insert sequence idx in a column of matrix X
        if mx < len(x[0][idx]): X[:mx, idx] = x[0][idx][:mx]
        else: X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx: X[len(x[0][idx]):, idx] = state['null_sym_source']

        # Initialize Xmask column with ones in all positions that were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        if len(x[0][idx]) < mx: Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in range(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my: Y[len(y[0][idx]):, idx] = state['null_sym_target']
        Ymask[:len(y[0][idx]), idx] = 1.
        if len(y[0][idx]) < my: Ymask[len(y[0][idx]), idx] = 1.

    null_inputs = numpy.zeros(X.shape[1])

    # We say that an input pair is valid if both:
    # - either source sequence or target sequence is non-empty
    # - source sequence and target sequence have null_sym ending
    # Why did not we filter them earlier?
    for idx in range(X.shape[1]):
        if numpy.sum(Xmask[:,idx]) == 0 and numpy.sum(Ymask[:,idx]) == 0:
            null_inputs[idx] = 1
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym_source']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym_target']:
            null_inputs[idx] = 1

    valid_inputs = 1. - null_inputs

    # Leave only valid inputs
    X = X[:,valid_inputs.nonzero()[0]]
    Y = Y[:,valid_inputs.nonzero()[0]]
    Xmask = Xmask[:,valid_inputs.nonzero()[0]]
    Ymask = Ymask[:,valid_inputs.nonzero()[0]]
    if len(valid_inputs.nonzero()[0]) <= 0: return None

    # Unknown words
    X[X >= state['n_sym_source']] = state['unk_sym_source']
    Y[Y >= state['n_sym_target']] = state['unk_sym_target']

    if return_dict: return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else: return X, Xmask, Y, Ymask

def get_batch_iterator(state):

    class Iterator(PytablesBitextIterator):
        def __init__(self, *args, **kwargs):
            PytablesBitextIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator.next(self) for k in range(k_batches)]
                x = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = numpy.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = numpy.asarray([list(map(len, x)), list(map(len, y))])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]], return_dict=True)
                    if batch: yield batch

        def next(self, peek=False):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            if not self.batch_iter: raise StopIteration
            batch = next(self.batch_iter)
            if peek: self.peeked_batch = batch
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'][0],
        source_file=state['source'][0],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        max_len=state['seqlen'])
    return train_data
   


def none_if_zero(x):
    if x == 0: return None
    return x

def prefix_lookup(state, p, s):
    if '%s_%s'%(p,s) in state: return state['%s_%s'%(p, s)]
    return state[s]


def parse_input(state, word2idx, line, raise_unk=False, idx2word=None, unk_sym=-1, null_sym=-1):
    """
    parse a string sentence into a sequence of word indices
    :param word2idx: a dictionary of [word,index]
    :param line: input sentence
    """
    if unk_sym < 0: unk_sym = state['unk_sym_source']
    if null_sym < 0: null_sym = state['null_sym_source']
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= state['n_sym_source']:
            seq[idx] = unk_sym
        if seq[idx] == unk_sym and raise_unk:
            raise Exception("Unknown word {}".format(sx))

    seq[-1] = null_sym
    if idx2word:
        idx2word[null_sym] = '</s>'
        idx2word[unk_sym] = state['oov']
        parsed_in = [idx2word[sx] for sx in seq]
        return seq, " ".join(parsed_in)

    return seq, seqin
