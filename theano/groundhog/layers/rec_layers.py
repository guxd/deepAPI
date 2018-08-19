"""
Recurrent layers.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import copy
import theano
import theano.tensor as TT
# Nicer interface of scan
from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import logging
from groundhog import utils
from groundhog.utils import sample_weights, \
        sample_weights_classic,\
        sample_weights_orth, \
        init_bias, \
        constant_shape, \
        sample_zeros
from .ff_layers import ReplicateLayer
from .basic import Layer

logger = logging.getLogger(__name__)


class RecurrentLayer(Layer):
    """
        Standard recurrent layer with gates.
        See arXiv verion of our paper.
    """
    def __init__(self, rng,
                 n_hids=500,
                 scale=.01,
                 sparsity = -1,
                 activation = TT.tanh,
                 activ_noise=0.,
                 weight_noise=False,
                 bias_fn='init_bias',
                 bias_scale = 0.,
                 dropout = 1.,
                 init_fn='sample_weights',
                 kind_reg = None,
                 grad_scale = 1.,
                 profile = 0,
                 gating = False,
                 reseting = False,
                 gater_activation = TT.nnet.sigmoid,
                 reseter_activation = TT.nnet.sigmoid,
                 name=None):
        logger.debug("RecurrentLayer is used")
        """
        :type rng: numpy random generator
        :param rng: numpy random generator

        :type n_in: int
        :param n_in: number of inputs units

        :type n_hids: int
        :param n_hids: Number of hidden units on each layer of the MLP

        :type activation: string/function or list of
        :param activation: Activation function for the embedding layers. If
            a list it needs to have a value for each layer. If not, the same
            activation will be applied to all layers

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


        :type weight_noise: bool
        :param weight_noise: If true, the model is used with weight noise
            (and the right shared variable are constructed, to keep track of the
            noise)

        :type dropout: float
        :param dropout: the probability with which hidden units are dropped
            from the hidden layer. If set to 1, dropout is not used

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

        :type grad_scale: float or theano scalar
        :param grad_scale: factor with which the gradients with respect to
            the parameters of this layer are scaled. It is used for
            differentiating between the different parameters of a model.

        :type gating: bool
        :param gating: If true, an update gate is used

        :type reseting: bool
        :param reseting: If true, a reset gate is used

        :type gater_activation: string or function
        :param name: The activation function of the update gate

        :type reseter_activation: string or function
        :param name: The activation function of the reset gate

        :type name: string
        :param name: name of the layer (used to name parameters). NB: in
            this library names are very important because certain parts of the
            code relies on name to disambiguate between variables, therefore
            each layer should have a unique name.

        """
        self.grad_scale = grad_scale

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(gater_activation) is str or type(gater_activation) is unicode:
            gater_activation = eval(gater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.sparsity = sparsity
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.weight_noise = weight_noise
        self.activ_noise = activ_noise
        self.profile = profile
        self.dropout = dropout
        self.gating = gating
        self.reseting = reseting
        self.gater_activation = gater_activation
        self.reseter_activation = reseter_activation

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayer, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.trng = RandomStreams(self.rng.randint(int(1e6)))
        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(
                self.init_fn(self.n_hids,
                self.n_hids,
                self.sparsity,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        if self.gating:
            self.G_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="G_%s"%self.name)
            self.params.append(self.G_hh)
        if self.reseting:
            self.R_hh = theano.shared(
                    self.init_fn(self.n_hids,
                    self.n_hids,
                    self.sparsity,
                    self.scale,
                    rng=self.rng),
                    name="R_%s"%self.name)
            self.params.append(self.R_hh)
        self.params_grad_scale = [self.grad_scale for x in self.params]
        self.restricted_params = [x for x in self.params]
        if self.weight_noise:
            self.nW_hh = theano.shared(self.W_hh.get_value()*0, name='noise_'+self.W_hh.name)
            self.noise_params = [self.nW_hh]
            if self.gating:
                self.nG_hh = theano.shared(self.G_hh.get_value()*0, name='noise_'+self.G_hh.name)
                self.noise_params += [self.nG_hh]
            if self.reseting:
                self.nR_hh = theano.shared(self.R_hh.get_value()*0, name='noise_'+self.R_hh.name)
                self.noise_params += [self.nR_hh]
            self.noise_params_shape_fn = [constant_shape(x.get_value().shape)
                            for x in self.noise_params]

    def step_fprop(self,
                   state_below,
                   mask = None,
                   state_before = None,
                   gater_below = None,
                   reseter_below = None,
                   use_noise=True,
                   no_noise_bias = False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a
            minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the
            layer

        :type gater_below: theano variable
        :param gater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        rval = []
        if self.weight_noise and use_noise and self.noise_params:
            W_hh = self.W_hh + self.nW_hh
            if self.gating:
                G_hh = self.G_hh + self.nG_hh
            if self.reseting:
                R_hh = self.R_hh + self.nR_hh
        else:
            W_hh = self.W_hh
            if self.gating:
                G_hh = self.G_hh
            if self.reseting:
                R_hh = self.R_hh

        # Reset gate:
        # optionally reset the hidden state.
        if self.reseting and reseter_below:
            reseter = self.reseter_activation(TT.dot(state_before, R_hh) +
                    reseter_below)
            reseted_state_before = reseter * state_before
        else:
            reseted_state_before = state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        preactiv = TT.cast(preactiv, theano.config.floatX)
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        if self.gating and gater_below:
            gater = self.gater_activation(TT.dot(state_before, G_hh) +
                    gater_below)
            h = gater * h + (1-gater) * state_before

        if self.activ_noise and use_noise:
            h = h + self.trng.normal(h.shape, avg=0, std=self.activ_noise, dtype=h.dtype)
        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before
        return h

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias = False
             ):
        """
        Evaluates the forward through a recurrent layer

        :type state_below: theano variable
        :param state_below: the input of the recurrent layer (previous state as well as the new input?)

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a minibatch

        :type init_state: theano variable or None
        :param init_state: initial state for the hidden layer

        :type n_steps: None or int or theano scalar
        :param n_steps: Number of steps the recurrent network does (i.e., sequence length)

        :type batch_size: int
        :param batch_size: the size of the minibatch over which scan runs

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type truncate_gradient: int
        :param truncate_gradient: If negative, no truncation is used,
            otherwise truncated BPTT is used, where you go backwards only this
            amount of steps

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if gater_below:
                gater_below = gater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        # FIXME: Find a way to clean this up
        if self.reseting and reseter_below:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below, reseter_below]
                    fn = lambda x,y,g,r,z : self.step_fprop(x,y,z, gater_below=g, reseter_below=r, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below, reseter_below]
                    fn = lambda tx, tg,tr, ty: self.step_fprop(tx, None, ty, gater_below=tg,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask, reseter_below]
                    fn = lambda x,y,r,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                                        reseter_below=r,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, reseter_below]
                    fn = lambda tx,tr,ty: self.step_fprop(tx, None, ty,
                                                        reseter_below=tr,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
        else:
            if self.gating and gater_below:
                if mask:
                    inps = [state_below, mask, gater_below]
                    fn = lambda x,y,g,z : self.step_fprop(x,y,z, gater_below=g, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below, gater_below]
                    fn = lambda tx, tg, ty: self.step_fprop(tx, None, ty, gater_below=tg,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)
            else:
                if mask:
                    inps = [state_below, mask]
                    fn = lambda x,y,z : self.step_fprop(x,y,z, use_noise=use_noise,
                                                       no_noise_bias=no_noise_bias)
                else:
                    inps = [state_below]
                    fn = lambda tx, ty: self.step_fprop(tx, None, ty,
                                                        use_noise=use_noise,
                                                        no_noise_bias=no_noise_bias)

        rval, updates = theano.scan(fn,
                        sequences = inps,
                        outputs_info = [init_state],
                        name='layer_%s'%self.name,
                        profile=self.profile,
                        truncate_gradient = truncate_gradient,
                        n_steps = nsteps)
        new_h = rval
        self.out = rval
        self.rval = rval
        self.updates =updates

        return self.out


class RecurrentLayerWithSearch(Layer):
    """ 
    Attention GRU - Attention based recurrent layer with gates.
    """
    def __init__(self, rng,
                 n_hids,
                 c_dim=None,#dimension of context vectors
                 scale=.01,
                 activation=TT.tanh,
                 bias_fn='init_bias',
                 bias_scale=0.,
                 init_fn='sample_weights',
                 gating=False,
                 reseting=False,
                 dropout=1.,
                 gater_activation=TT.nnet.sigmoid,
                 reseter_activation=TT.nnet.sigmoid,
                 weight_noise=False,
                 name=None):
        logger.debug("RecurrentLayerWithSearch is used")

        self.grad_scale = 1
        assert gating == True
        assert reseting == True
        assert dropout == 1.
        assert weight_noise == False
        updater_activation = gater_activation

        if type(init_fn) is str or type(init_fn) is unicode:
            init_fn = eval(init_fn)
        if type(bias_fn) is str or type(bias_fn) is unicode:
            bias_fn = eval(bias_fn)
        if type(activation) is str or type(activation) is unicode:
            activation = eval(activation)
        if type(updater_activation) is str or type(updater_activation) is unicode:
            updater_activation = eval(updater_activation)
        if type(reseter_activation) is str or type(reseter_activation) is unicode:
            reseter_activation = eval(reseter_activation)

        self.scale = scale
        self.activation = activation
        self.n_hids = n_hids
        self.bias_scale = bias_scale
        self.bias_fn = bias_fn
        self.init_fn = init_fn
        self.updater_activation = updater_activation
        self.reseter_activation = reseter_activation
        self.c_dim = c_dim

        assert rng is not None, "random number generator should not be empty!"

        super(RecurrentLayerWithSearch, self).__init__(self.n_hids,
                self.n_hids, rng, name)

        self.params = []
        self._init_params()

    def _init_params(self):
        self.W_hh = theano.shared(#W_hh: weights denote how much current input should be 
                                  #used to update the hidden state
                self.init_fn(self.n_hids,
                self.n_hids,
                -1,
                self.scale,
                rng=self.rng),
                name="W_%s"%self.name)
        self.params = [self.W_hh]
        self.G_hh = theano.shared(#G_hh: weights denote how much previous hidden state(h)
                                  #should be updated 
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="G_%s"%self.name)
        self.params.append(self.G_hh)
        self.R_hh = theano.shared(#R_hh: weights denote how much previous state should be reset
                self.init_fn(self.n_hids,
                    self.n_hids,
                    -1,
                    self.scale,
                    rng=self.rng),
                name="R_%s"%self.name)
        self.params.append(self.R_hh)
        self.A_cp = theano.shared(#A_cp [attention alignments?]:
                                    # weights from context(c) to projection(p)
                sample_weights_classic(self.c_dim,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="A_%s"%self.name)
        self.params.append(self.A_cp)
        self.B_hp = theano.shared(#B_hp: weights from previous hidden(h) to projection(p)
                sample_weights_classic(self.n_hids,
                    self.n_hids,
                    -1,
                    10 ** (-3),
                    rng=self.rng),
                name="B_%s"%self.name)
        self.params.append(self.B_hp)
        self.D_pe = theano.shared(#D_pe: weights from projection(p) to energy(e)
                numpy.zeros((self.n_hids, 1), dtype="float32"),
                name="D_%s"%self.name)
        self.params.append(self.D_pe)
        self.params_grad_scale = [self.grad_scale for x in self.params]
       
    def set_decoding_layers(self, c_inputer, c_reseter, c_updater):
        self.c_inputer = c_inputer
        self.c_reseter = c_reseter
        self.c_updater = c_updater
        for layer in [c_inputer, c_reseter, c_updater]:
            self.params += layer.params
            self.params_grad_scale += layer.params_grad_scale

    def step_fprop(self,
                   state_below,
                   state_before,
                   gater_below=None,
                   reseter_below=None,
                   mask=None,
                   c=None,
                   c_mask=None,
                   p_from_c=None,
                   use_noise=True,
                   no_noise_bias=False,
                   step_num=None,
                   return_alignment=False):
        """
        Constructs the computational graph of this layer.

        :type state_below: theano variable
        :param state_below: the input to the layer

        :type mask: None or theano variable
        :param mask: mask describing the length of each sequence in a minibatch

        :type state_before: theano variable
        :param state_before: the previous value of the hidden state of the layer

        :type updater_below: theano variable
        :param updater_below: the input to the update gate

        :type reseter_below: theano variable
        :param reseter_below: the input to the reset gate

        :type use_noise: bool
        :param use_noise: flag saying if weight noise should be used in
            computing the output of this layer

        :type no_noise_bias: bool
        :param no_noise_bias: flag saying if weight noise should be added to
            the bias as well
        """

        updater_below = gater_below

        W_hh = self.W_hh
        G_hh = self.G_hh
        R_hh = self.R_hh
        A_cp = self.A_cp
        B_hp = self.B_hp
        D_pe = self.D_pe

        # The code works only with 3D tensors
        cndim = c.ndim
        if cndim == 2:
            c = c[:, None, :]

        # Warning: either source_num or target_num should be equal,
        #          or one of them sould be 1 (they have to broadcast)
        #          for the following code to make any sense.
        source_len = c.shape[0]#sequence length
        source_num = c.shape[1]#number of sequences in a batch 
        target_num = state_before.shape[0]
        dim = self.n_hids

        # Form projection to the tanh layer from the previous hidden state
        # Shape: (source_len, target_num, dim)
        p_from_h = ReplicateLayer(source_len)(utils.dot(state_before, B_hp)).out

        # Form projection to the tanh layer from the source annotation.
        if not p_from_c:
            p_from_c =  utils.dot(c, A_cp).reshape((source_len, source_num, dim))

        # Sum projections - broadcasting happens at the dimension 1.
        p = p_from_h + p_from_c

        # Apply non-linearity and project to energy.
        energy = TT.exp(utils.dot(TT.tanh(p), D_pe)).reshape((source_len, target_num))
        if c_mask:
            # This is used for batches only, that is target_num == source_num
            energy *= c_mask

        # Calculate energy sums.
        normalizer = energy.sum(axis=0)

        # Get probabilities. (softmax?)
        probs = energy / normalizer 

        # Calculate weighted sums of source annotations.
        # If target_num == 1, c shoulds broadcasted at the 1st dimension.
        # Probabilities are broadcasted at the 2nd dimension.
        ctx = (c * probs.dimshuffle(0, 1, 'x')).sum(axis=0)#averaged context (see the picture)

        state_below += self.c_inputer(ctx).out
        reseter_below += self.c_reseter(ctx).out
        updater_below += self.c_updater(ctx).out

        # Reset gate:
        # optionally reset the hidden state.
        reseter = self.reseter_activation(TT.dot(state_before, R_hh)+reseter_below)
        reseted_state_before = reseter * state_before

        # Feed the input to obtain potential new state.
        preactiv = TT.dot(reseted_state_before, W_hh) + state_below
        h = self.activation(preactiv)

        # Update gate:
        # optionally reject the potential new state and use the new one.
        updater = self.updater_activation(TT.dot(state_before, G_hh) + updater_below)
        h = updater * h + (1-updater) * state_before #h_t=z*h_{t-1}+(1-z)*h_t

        if mask is not None:
            if h.ndim ==2 and mask.ndim==1:
                mask = mask.dimshuffle(0,'x')
            h = mask * h + (1-mask) * state_before

        results = [h, ctx]
        if return_alignment:
            results += [probs]
        return results

    def fprop(self,
              state_below,
              mask=None,
              init_state=None,
              gater_below=None,
              reseter_below=None,
              c=None,
              c_mask=None,
              nsteps=None,
              batch_size=None,
              use_noise=True,
              truncate_gradient=-1,
              no_noise_bias=False,
              return_alignment=False):

        updater_below = gater_below

        if theano.config.floatX=='float32':
            floatX = numpy.float32
        else:
            floatX = numpy.float64
        if nsteps is None:
            nsteps = state_below.shape[0]
            if batch_size and batch_size != 1:
                nsteps = nsteps / batch_size
        if batch_size is None and state_below.ndim == 3:
            batch_size = state_below.shape[1]
        if state_below.ndim == 2 and \
           (not isinstance(batch_size,int) or batch_size > 1):
            state_below = state_below.reshape((nsteps, batch_size, self.n_in))
            if updater_below:
                updater_below = updater_below.reshape((nsteps, batch_size, self.n_in))
            if reseter_below:
                reseter_below = reseter_below.reshape((nsteps, batch_size, self.n_in))

        if not init_state:
            if not isinstance(batch_size, int) or batch_size != 1:
                init_state = TT.alloc(floatX(0), batch_size, self.n_hids)
            else:
                init_state = TT.alloc(floatX(0), self.n_hids)

        p_from_c =  utils.dot(c, self.A_cp).reshape(#attention weights for each hidden state
                (c.shape[0], c.shape[1], self.n_hids))
        
        if mask:
            sequences = [state_below, mask, updater_below, reseter_below]
            non_sequences = [c, c_mask, p_from_c] 
            #              seqs    | out |  non_seqs
            fn = lambda x, m, g, r,   h,   c1, cm, pc : self.step_fprop(x, h, mask=m,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc, c_mask=cm,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)
        else:
            sequences = [state_below, updater_below, reseter_below]
            non_sequences = [c, p_from_c]
            #            seqs   | out | non_seqs
            fn = lambda x, g, r,   h,    c1, pc : self.step_fprop(x, h,
                    gater_below=g, reseter_below=r,
                    c=c1, p_from_c=pc,
                    use_noise=use_noise, no_noise_bias=no_noise_bias,
                    return_alignment=return_alignment)

        outputs_info = [init_state, None]
        if return_alignment:
            outputs_info.append(None)

        #use scan to repetitively update hidden states
        rval, updates = theano.scan(fn,
                        sequences=sequences,
                        non_sequences=non_sequences,
                        outputs_info=outputs_info,
                        name='layer_%s'%self.name,
                        truncate_gradient=truncate_gradient,
                        n_steps=nsteps)
        self.out = rval
        self.rval = rval
        self.updates = updates

        return self.out   
 
