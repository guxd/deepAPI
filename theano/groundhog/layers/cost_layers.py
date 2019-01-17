"""
Cost layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import copy
import logging
import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pickle

from groundhog import utils
from groundhog.utils import sample_weights, sample_weights_classic,\
    init_bias, constant_shape, sample_zeros

from .basic import Layer

logger = logging.getLogger(__name__)

class CostLayer(Layer):
    """
    Base class for all cost layers
    """
    def __init__(self, rng,
                 n_in,
                 n_out,
                 scale,
                 sparsity,
                 rank_n_approx=0,
                 rank_n_activ='lambda x: x',
                 weight_noise=False,
                 init_fn='sample_weights_classic',
                 bias_fn='init_bias',
                 bias_scale=0.,
                 sum_over_time=True,#whether sum the cost of a whole sentence, can be set in Decoder._create_readout_layers() at encdec.py
                 additional_inputs=None,
                 grad_scale=1.,
                 use_nce=False,
                 name=None):
        """
        :type rng: numpy random generator
        :param rng: numpy random generator used to sample weights

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units

        :type scale: float or list of
        :param scale: depending on the initialization function, it can be
            the standard deviation of the Gaussian from which the weights
            are sampled or the largest singular value. If a single value it
            will be used for each layer, otherwise it has to have one value
            for each layer

        :type sparsity: int or list of
        :param sparsity: if a single value, it will be used for each layer,
            otherwise it has to be a list with as many values as layers. If
            negative, it means the weight matrix is dense. Otherwise it
            means this many randomly selected input units are connected to
            an output unit

        :type rank_n_approx: int
        :param rank_n_approx: It applies to the first layer only. If
            positive and larger than 0, the first weight matrix is
            factorized into two matrices. The first one goes from input to
            `rank_n_approx` hidden units, the second from `rank_n_approx` to
            the number of units on the second layer

        :type rank_n_activ: string or function
        :param rank_n_activ: Function that is applied on on the intermediary
            layer formed from factorizing the first weight matrix (Q: do we
            need this?)

        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track
            of the noise)

        :type init_fn: string or function
        :param init_fn: function used to initialize the weights of the
            layer. We recommend using either `sample_weights_classic` or
            `sample_weights` defined in the utils

        :type bias_fn: string or function
        :param bias_fn: function used to initialize the biases. We recommend
            using `init_bias` defined in the utils

        :type bias_scale: float
        :param bias_scale: argument passed to `bias_fn`, depicting the scale
            of the initial bias

        :type sum_over_time: bool
        :param sum_over_time: flag, stating if, when computing the cost, we
            should take the sum over time, or the mean. If you have variable
            length sequences, please take the sum over time

        :type additional_inputs: None or list of ints
        :param additional_inputs: dimensionality of each additional input

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type use_nce: bool
        :param use_nce: flag, if true, do not use MLE, but NCE-like cost

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.
        """
        self.grad_scale = grad_scale
        assert rank_n_approx >= 0, "Please enter a valid rank_n_approx"
        self.rank_n_approx = rank_n_approx
        if type(rank_n_activ) is str:
            rank_n_activ = eval(rank_n_activ)
        self.rank_n_activ = rank_n_activ
        super(CostLayer, self).__init__(n_in, n_out, rng, name)
        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.scale = scale
        if isinstance(bias_fn, str):
            self.bias_fn = eval(bias_fn)
        else:
            self.bias_fn = bias_fn
        self.bias_scale = bias_scale
        self.sum_over_time = sum_over_time
        self.weight_noise = weight_noise
        self.sparsity = sparsity
        if self.sparsity < 0:
            self.sparsity = n_out
        if type(init_fn) is str:
            init_fn = eval(init_fn)
        self.init_fn = init_fn
        self.additional_inputs = additional_inputs
        self.use_nce = use_nce
        self._init_params()

    def _init_params(self):
        """
        Initialize the parameters of the layer, either by using sparse
        initialization or small isotropic noise.
        """
        if self.rank_n_approx:
            W_em1 = self.init_fn(self.n_in,
                                 self.rank_n_approx,
                                 self.sparsity,
                                 self.scale,
                                 self.rng)
            W_em2 = self.init_fn(self.rank_n_approx,
                                 self.n_out,
                                 self.sparsity,
                                 self.scale,
                                 self.rng)
            self.W_em1 = theano.shared(W_em1,
                                       name='W1_%s' % self.name)
            self.W_em2 = theano.shared(W_em2,
                                       name='W2_%s' % self.name)
            self.b_em = theano.shared(
                self.bias_fn(self.n_out, self.bias_scale, self.rng),
                name='b_%s' % self.name)
            self.params += [self.W_em1, self.W_em2, self.b_em]

            if self.weight_noise:
                self.nW_em1 = theano.shared(W_em1*0.,
                                            name='noise_W1_%s' % self.name)
                self.nW_em2 = theano.shared(W_em*0.,
                                            name='noise_W2_%s' % self.name)
                self.nb_em = theano.shared(b_em*0.,
                                           name='noise_b_%s' % self.name)
                self.noise_params = [self.nW_em1, self.nW_em2, self.nb_em]
                self.noise_params_shape_fn = [
                    constant_shape(x.get_value().shape)
                    for x in self.noise_params]

        else:
            W_em = self.init_fn(self.n_in, self.n_out, self.sparsity, self.scale, self.rng)
            self.W_em = theano.shared(W_em, name='W_%s' % self.name)
            self.b_em = theano.shared(
                self.bias_fn(self.n_out, self.bias_scale, self.rng),
                name='b_%s' % self.name)

            self.params += [self.W_em, self.b_em]
            if self.weight_noise:
                self.nW_em = theano.shared(W_em*0., name='noise_W_%s' % self.name)
                self.nb_em = theano.shared(numpy.zeros((self.n_out,), dtype=theano.config.floatX),
                    name='noise_b_%s' % self.name)
                self.noise_params = [self.nW_em, self.nb_em]
                self.noise_params_shape_fn = [constant_shape(x.get_value().shape) for x in self.noise_params]
        self.additional_weights = []
        self.noise_additional_weights = []
        if self.additional_inputs:
            for pos, size in enumerate(self.additional_inputs):
                W_add = self.init_fn(size, self.n_out, self.sparsity, self.scale, self.rng)
                self.additional_weights += [theano.shared(W_add, name='W_add%d_%s'%(pos, self.name))]
                if self.weight_noise:
                    self.noise_additional_weights += [
                        theano.shared(W_add*0., name='noise_W_add%d_%s'%(pos, self.name))]
        self.params = self.params + self.additional_weights
        self.noise_params += self.noise_additional_weights
        self.noise_params_shape_fn += [
            constant_shape(x.get_value().shape)
            for x in self.noise_additional_weights]

        self.params_grad_scale = [self.grad_scale for x in self.params]

    def compute_sample(self, state_below, temp=1, use_noise=False):
        """
        Constructs the theano expression that samples from the output layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model
        """
        raise NotImplemented

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=None,
                 use_noise=True,
                 additional_inputs=None,
                 no_noise_bias=False):
        """
        Computes the expression of the cost of the model (given the type of
        layer used).

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type target: tensor or layer
        :param target: The theano expression (or groundhog layer)
            representing the target (used to evaluate the prediction of the
            output layer)

        :type mask: None or mask or layer
        :param mask: Mask, depicting which of the predictions should be
            ignored (e.g. due to them resulting from padding a sequence
            with 0s)

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type reg: None or layer or theano scalar expression
        :param reg: additional regularization term that should be added to
            the cost

        :type scale: float or None or theano scalar
        :param scale: scaling factor with which the cost is multiplied

        :type sum_over_time: bool or None
        :param sum_over_time: this flag overwrites the value given to this
            property in the constructor of the class

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type additional_inputs: list theano variable or layers
        :param additional_inputs: list of theano variables or layers
            representing the additional inputs

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        raise NotImplemented

    def get_grads(self,
                  state_below,
                  target=None,
                  mask=None,
                  temp=1,
                  reg=None,
                  scale=None,
                  additional_gradients=None,
                  sum_over_time=None,
                  use_noise=True,
                  additional_inputs=None,
                  no_noise_bias=False):
        """
        Computes the expression of the gradients of the cost with respect to
        all parameters of the model.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type target: tensor or layer
        :param target: The theano expression (or groundhog layer)
            representing the target (used to evaluate the prediction of the output layer)

        :type mask: None or mask or layer
        :param mask: Mask, depicting which of the predictions should be
            ignored (e.g. due to them resulting from padding a sequence with 0s)

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type reg: None or layer or theano scalar expression
        :param reg: additional regularization term that should be added to the cost

        :type scale: float or None or theano scalar
        :param scale: scaling factor with which the cost is multiplied

        :type additional_gradients: list of tuples of the form
            (param, gradient)
        :param additional_gradiens: A list of tuples. Each tuple has as its
            first element the parameter, and as second element a gradient
            expression that should be added to the gradient resulting from the
            cost. Not all parameters need to have an additional gradient.

        :type sum_over_time: bool or None
        :param sum_over_time: this flag overwrites the value given to this
            property in the constructor of the class

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        cost = self.get_cost(state_below,
                             target,
                             mask=mask,
                             reg=reg,
                             scale=scale,
                             sum_over_time=sum_over_time,
                             use_noise=use_noise,
                             additional_inputs=additional_inputs,
                             no_noise_bias=no_noise_bias)
        
        '''Put your modification of cost here..'''
        

        
        
        
        
        logger.debug("Get grads")
        grads = TT.grad(cost.mean(), self.params,disconnected_inputs='warn'
                        )
        logger.debug("Got grads")
        if additional_gradients:
            for p, gp in additional_gradients:
                if p in self.params:
                    grads[self.params.index(p)] += gp
        if self.additional_gradients:
            for new_grads, to_replace, properties in self.additional_gradients:
                gparams, params = new_grads
                prop_expr = [x[1] for x in properties]
                replace = [(x[0], TT.grad(cost, x[1])) for x in to_replace]
                rval = theano.clone(gparams + prop_expr,
                                    replace=replace)
                gparams = rval[:len(gparams)]
                prop_expr = rval[len(gparams):]
                self.properties += [(x[0], y)
                                    for x, y in zip(properties, prop_expr)]
                for gp, p in zip(gparams, params):
                    grads[self.params.index(p)] += gp

        self.cost = cost
        self.grads = grads
        
        return cost, grads

    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        Sample a sequence from the model `model` whose output layer is given by `self`.

        :type model: groundhog model class
        :param model: model that has `self` as its output layer

        :type length: int
        :param length: length of the sequence to sample

        :type temp: float
        :param temp: temperature to use during sampling
        """
        raise NotImplemented



class SigmoidLayer(CostLayer):
    """
    Sigmoid output layer.
    """
    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        See parent class.
        """
        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs

        character_level = False
        if hasattr(model, 'character_level'):
            character_level = model.character_level
        if model.del_noise: model.del_noise()
        [c,values, probs] = model.sample_fn(length, temp, *inps)# get samples (c is the context vector of source sentence)
        # Assumes values matrix
        #print 'Generated sample is:'
        #print
        if values.ndim > 1:
            for d in xrange(2):
                print ('%d-th sentence' % d)
                print ('Input: ',)
                if character_level:
                    sen = []
                    for k in xrange(inps[0].shape[0]):
                        if model.word_indxs_src[inps[0][k][d]] == '</s>': break
                        sen.append(model.word_indxs_src[inps[0][k][d]])
                    print ("".join(sen),)
                else:
                    for k in xrange(inps[0].shape[0]):
                        print (model.word_indxs_src[inps[0][k][d]],)
                        if model.word_indxs_src[inps[0][k][d]] == '</s>': break
                print ('')
                print ('Output: ',)
                if character_level:
                    sen = []
                    for k in xrange(values.shape[0]):
                        if model.word_indxs[values[k][d]] == '</s>': break
                        sen.append(model.word_indxs[values[k][d]])
                    print ("".join(sen),)
                else:
                    for k in xrange(values.shape[0]):
                        print (model.word_indxs[values[k][d]],)
                        if model.word_indxs[values[k][d]] == '</s>': break
                print()
                print()
        else:
            print ('Input:  ',)
            if character_level:
                sen = []
                for k in xrange(inps[0].shape[0]):
                    if model.word_indxs_src[inps[0][k]] == '</s>':
                        break
                    sen.append(model.word_indxs_src[inps[0][k]])
                print ("".join(sen),)
            else:
                for k in xrange(inps[0].shape[0]):
                    print (model.word_indxs_src[inps[0][k]],)
                    if model.word_indxs_src[inps[0][k]] == '</s>':
                        break
            print ('')
            print ('Output: ',)
            if character_level:
                sen = []
                for k in xrange(values.shape[0]):
                    if model.word_indxs[values[k]] == '</s>':
                        break
                    sen.append(model.word_indxs[values[k]])
                print ("".join(sen)),
            else:
                for k in xrange(values.shape[0]):
                    print (model.word_indxs[values[k]],)
                    if model.word_indxs[values[k]] == '</s>':
                        break
            print()
            print()

    def fprop(self,
              state_below,
              temp=numpy.float32(1),
              use_noise=True,
              additional_inputs=None,
              no_noise_bias=False):
        """
        Forward pass through the cost layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """

        if self.rank_n_approx:
            if use_noise and self.noise_params:
                emb_val = self.rank_n_activ(utils.dot(state_below,
                                                      self.W_em1+self.nW_em1))
                emb_val = TT.dot(self.W_em2 + self.nW_em2, emb_val)
            else:
                emb_val = self.rank_n_activ(utils.dot(state_below, self.W_em1))
                emb_val = TT.dot(self.W_em2, emb_val)
        else:
            if use_noise and self.noise_params:
                emb_val = utils.dot(state_below, self.W_em + self.nW_em)
            else:
                emb_val = utils.dot(state_below, self.W_em)

        if additional_inputs:
            if use_noise and self.noise_params:
                for inp, weight, noise_weight in zip(
                    additional_inputs, self.additional_weights,
                    self.noise_additional_weights):
                    emb_val += utils.dot(inp, (noise_weight + weight))
            else:
                for inp, weight in zip(additional_inputs, self.additional_weights):
                    emb_val += utils.dot(inp, weight)
        self.preactiv = emb_val
        if use_noise and self.noise_params and not no_noise_bias:
            emb_val = TT.nnet.sigmoid(temp *
                                      (emb_val + self.b_em + self.nb_em))
        else:
            emb_val = TT.nnet.sigmoid(temp * (emb_val + self.b_em))
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def compute_sample(self,
                       state_below,
                       temp=1,
                       additional_inputs=None,
                       use_noise=False):
        """
        See parent class.
        """
        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 additional_inputs=additional_inputs,
                                 use_noise=use_noise)
        pvals = class_probs
        if pvals.ndim == 1:
            pvals = pvals.dimshuffle('x', 0)
        sample = self.trng.binomial(pvals.shape, p=pvals,
                                    dtype='int64')
        if class_probs.ndim == 1:
            sample = sample[0]
        self.sample = sample
        return sample

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,
                 scale=None,
                 sum_over_time=None,
                 use_noise=True,
                 additional_inputs=None,
                 no_noise_bias=False):
        """
        See parent class
        """
        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 use_noise=use_noise,
                                 additional_inputs=additional_inputs,
                                 no_noise_bias=no_noise_bias)
        pvals = class_probs
        assert target, 'Computing the cost requires a target'
        if target.ndim == 3:
            target = target.reshape((target.shape[0]*target.shape[1],
                                    target.shape[2]))
        assert 'float' in target.dtype
        # Do we need the safety net of 1e-12  ?
        cost = -TT.log(TT.maximum(1e-12, class_probs)) * target -\
            TT.log(TT.maximum(1e-12, 1 - class_probs)) * (1 - target)
        print ('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        if cost.ndim > 1:
            cost = cost.sum(1)
        if mask:
            mask = mask.flatten()
            cost = cost * TT.cast(mask, theano.config.floatX)
        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim == 3:
                sh0 = TT.cast(state_below.shape[0],
                              theano.config.floatX)
                sh1 = TT.cast(state_below.shape[1],
                              theano.config.floatX)
                self.cost = cost.sum()/sh1
            else:
                self.cost = cost.sum()
        else:
            self.cost = cost.mean()
        if scale:
            self.cost = self.cost*scale
        if reg:
            self.cost = self.cost + reg
        self.out = self.cost
        self.mask = mask
        self.cost_scale = scale
        return self.cost


class SoftmaxLayer(CostLayer):
    """
    Softmax output layer.
    """

    def _get_samples(self, model, length=30, temp=1, *inps):
        """
        See parent class
        """
        if not hasattr(model, 'word_indxs_src'):
            model.word_indxs_src = model.word_indxs

        character_level = False
        if hasattr(model, 'character_level'):
            character_level = model.character_level
        if model.del_noise:
            model.del_noise()
        [c, values, probs] = model.sample_fn(length, temp, *inps)# get samples (c is the context vector of source sentence)
        #print 'Generated sample is:'
        #print
        if values.ndim > 1:
            for d in range(2):
                print ('%d-th sentence' % d)
                print ('Input: ',end="")
                if character_level:
                    sen = []
                    for k in range(inps[0].shape[0]):
                        if model.word_indxs_src[inps[0][k][d]] == '</s>': break
                        sen.append(model.word_indxs_src[inps[0][k][d]])
                    print ("".join(sen),end="")
                else:
                    for k in range(inps[0].shape[0]):
                        print (model.word_indxs_src[inps[0][k][d]],end="")
                        if model.word_indxs_src[inps[0][k][d]] == '</s>': break
                print ('')
                print ('Output: ', end="")
                if character_level:
                    sen = []
                    for k in range(values.shape[0]):
                        if model.word_indxs[values[k][d]] == '</s>': break
                        sen.append(model.word_indxs[values[k][d]])
                    print ("".join(sen),end="")
                else:
                    for k in range(values.shape[0]):
                        print (model.word_indxs[values[k][d]],end="")
                        if model.word_indxs[values[k][d]] == '</s>': break
                print()
                print()
        else:
            print ('Input:  ',end="")
            if character_level:
                sen = []
                for k in range(inps[0].shape[0]):
                    if model.word_indxs_src[inps[0][k]] == '</s>': break
                    sen.append(model.word_indxs_src[inps[0][k]])
                print ("".join(sen),end=" ")
            else:
                for k in range(inps[0].shape[0]):
                    print (model.word_indxs_src[inps[0][k]],end=" ")
                    if model.word_indxs_src[inps[0][k]] == '</s>': break
            print ('')
            print ('Output: ',end="")
            if character_level:
                sen = []
                for k in range(values.shape[0]):
                    if model.word_indxs[values[k]] == '</s>': break
                    sen.append(model.word_indxs[values[k]])
                print ("".join(sen),end="")
            else:
                for k in range(values.shape[0]):
                    print (model.word_indxs[values[k]],end=" ")
                    if model.word_indxs[values[k]] == '</s>': break
            print()
            print()

    def fprop(self,
              state_below,
              temp=numpy.float32(1),
              use_noise=True,
              additional_inputs=None,
              no_noise_bias=False,
              target=None,
              full_softmax=True):
        """
        Forward pass through the cost layer.

        :type state_below: tensor or layer
        :param state_below: The theano expression (or groundhog layer)
            representing the input of the cost layer

        :type temp: float or tensor scalar
        :param temp: scalar representing the temperature that should be used
            when sampling from the output distribution

        :type use_noise: bool
        :param use_noise: flag. If true, noise is used when computing the
            output of the model

        :type no_noise_bias: bool
        :param no_noise_bias: flag, stating if weight noise should be added
            to the bias as well, or only to the weights
        """
        if not full_softmax:
            assert target != None, 'target must be given'
        if self.rank_n_approx:
            if self.weight_noise and use_noise and self.noise_params:
                emb_val = self.rank_n_activ(utils.dot(state_below,
                                                      self.W_em1+self.nW_em1))
                nW_em = self.nW_em2
            else:
                emb_val = self.rank_n_activ(utils.dot(state_below, self.W_em1))
            W_em = self.W_em2
        else:
            W_em = self.W_em
            if self.weight_noise:
                nW_em = self.nW_em
            emb_val = state_below

        if full_softmax:
            if self.weight_noise and use_noise and self.noise_params:
                emb_val = TT.dot(emb_val, W_em + nW_em)
            else:
                emb_val = TT.dot(emb_val, W_em)

            if additional_inputs:
                if use_noise and self.noise_params:
                    for inp, weight, noise_weight in zip(
                        additional_inputs, self.additional_weights,
                        self.noise_additional_weights):
                        emb_val += utils.dot(inp, (noise_weight + weight))
                else:
                    for inp, weight in zip(additional_inputs, self.additional_weights):
                        emb_val += utils.dot(inp, weight)
            if self.weight_noise and use_noise and self.noise_params and \
               not no_noise_bias:
                emb_val = temp * (emb_val + self.b_em + self.nb_em)
            else:
                emb_val = temp * (emb_val + self.b_em)
        else:
            W_em = W_em[:, target]
            if self.weight_noise:
                nW_em = nW_em[:, target]
                W_em += nW_em
            if emb_val.ndim == 3:
                emb_val = emb_val.reshape([emb_val.shape[0]*emb_val.shape[1], emb_val.shape[2]])
            emb_val = (W_em.T * emb_val).sum(1) + self.b_em[target]
            if self.weight_noise and use_noise:
                emb_val += self.nb_em[target]
            emb_val = temp * emb_val

        self.preactiv = emb_val
        if full_softmax:
            emb_val = utils.softmax(emb_val)
        else:
            emb_val = TT.nnet.sigmoid(emb_val)
        self.out = emb_val
        self.state_below = state_below
        self.model_output = emb_val
        return emb_val

    def compute_sample(self,
                       state_below,
                       temp=1,
                       use_noise=False,
                       additional_inputs=None):

        class_probs = self.fprop(state_below,
                                 temp=temp,
                                 additional_inputs=additional_inputs,
                                 use_noise=use_noise)
        pvals = class_probs
        if pvals.ndim == 1:
            pvals = pvals.dimshuffle('x', 0)
        sample = self.trng.multinomial(pvals=pvals, dtype='int64').argmax(axis=-1)
        if class_probs.ndim == 1: sample = sample[0]
        self.sample = sample
        return sample

    def get_cost(self,
                 state_below,
                 target=None,
                 mask=None,
                 temp=1,
                 reg=None,#extra regularization expression for cost
                 scale=None,
                 sum_over_time=False,
                 no_noise_bias=False,
                 additional_inputs=None,
                 use_noise=True):
        """
        Define cost function
        See parent class
        :param target:
            if mode == evaluation
                target sequences, matrix of word indices of shape (max_seq_len, batch_size),
                where each column is a sequence
            if mode != evaluation
                a vector of previous words of shape (n_samples,)

        :param mask:
            if mode == evaluation a 0/1 matrix determining lengths
                of the target sequences, must be None otherwise
        """

        def _grab_probs(class_probs, target):
            """
            get the probability of each word (index) in the target
            for example, given target of 3 sentences [[12,34,56],[1,33,1],[3,0,1],[0,0,0]] it returns prob of each word in the target
            :param class_probs:
                probabilities of all possible words
            :param target
                word indices of target sentence
            """
            shape0 = class_probs.shape[0]
            shape1 = class_probs.shape[1]
            target_ndim = target.ndim
            target_shape = target.shape
            if target.ndim > 1:
                target = target.flatten()
            assert target.ndim == 1, 'make sure target is a vector of ints'
            assert 'int' in target.dtype

            pos = TT.arange(shape0)*shape1
            new_targ = target + pos
            return class_probs.flatten()[new_targ]

        assert target, 'Computing the cost requires a target'
        target_ndim = target.ndim #dimemsion of the target , usually 2
        target_shape = target.shape#number of rows and columns, usually (max_seq_length,batch_size)
        

        if self.use_nce:#use noise contrastive estimation instead of maximize likelihood
            logger.debug("Using NCE")

            # positive samples: true targets
            class_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias,
                                     target=target.flatten(),
                                     full_softmax=False)
            # negative samples: a single uniform random sample per training sample
            negsamples = TT.cast(self.trng.uniform(class_probs.shape[0].reshape([1])) * self.n_out, 'int64')
            neg_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias,
                                     target=negsamples.flatten(),
                                     full_softmax=False)

            cost_target = class_probs
            cost_negsamples = 1. - neg_probs

            cost = -TT.log(cost_target)
            cost = cost - TT.cast(neg_probs.shape[0], theano.config.floatX) * TT.log(cost_negsamples)
        else: # use maximize likelihood
            class_probs = self.fprop(state_below,
                                     temp=temp,
                                     use_noise=use_noise,
                                     additional_inputs=additional_inputs,
                                     no_noise_bias=no_noise_bias)
            cost = -TT.log(_grab_probs(class_probs, target))#cost=-log p(y_i|x_i)

        self.word_probs = TT.exp(-cost.reshape(target_shape))
        
        if mask:# Set all the probs after the end-of-line to one
            self.word_probs = self.word_probs * mask + 1 - mask
        if mask:# Set all the costs after the end-of-line to 0
            cost = cost * TT.cast(mask.flatten(), theano.config.floatX)
        cost=cost.reshape(target_shape)   # A cost matrix recording costs for individual elements in the target matrix
        
         
            
            
            
            
            
        self.cost_per_sample = (cost.sum(axis=0)
                if target_ndim > 1
                else cost)#sum cost of each sentence (column)

        if sum_over_time is None:
            sum_over_time = self.sum_over_time
        if sum_over_time:
            if state_below.ndim == 3:
                cost = cost.reshape((state_below.shape[0],
                                     state_below.shape[1]))
                self.cost = cost.mean(1).sum()
            else:# for evaluation(log-likelihood graph), this branch is executed
                self.cost = TT.mean(self.cost_per_sample)#compute the average cost sum for each sentence
        else:#for sampling, this branch is executed
            if mask: self.cost=cost.sum()/mask.sum()#cost per_sequence_element
            else: self.cost = cost.mean()
        if scale: self.cost = self.cost*scale
        if reg: self.cost = self.cost + reg
            
        self.mask = mask
        self.cost_scale = scale
        
        return self.cost


