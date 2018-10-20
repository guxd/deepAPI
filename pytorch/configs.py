def config_Seq2Seq():
    conf = {
    'use_attention': False,
    'maxlen':30, # maximum sequence length

# Model Arguments
    'emb_size':200, # size of word embeddings
    'n_hidden':300, # number of hidden units per layer
    'n_layers':1, # number of layers
    'noise_radius':0.2, # stdev of noise for autoencoder (regularizer)
    'noise_anneal':0.995, # anneal noise_radius exponentially by this every 100 iterations
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)

# Training Arguments
    'batch_size':32,
    'epochs':30, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'lr_ae':1e-3, # autoencoder learning rate
    'beta1':0.9, # beta1 for adam
    'clip':5.0,  # gradient clipping, max norm        
    }
    return conf 

def config_DeepAPI():
    conf = {
    'use_attention': True,
    'maxlen':50, # maximum sequence length

# Model Arguments
    'emb_size':120, # size of word embeddings
    'n_hidden':1000, # number of hidden units per layer
    'n_layers':1, # number of layers
    'noise_radius':0.2, # stdev of noise for autoencoder (regularizer)
    'noise_anneal':0.995, # anneal noise_radius exponentially by this every 100 iterations
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)

# Training Arguments
    'batch_size':100,
    'epochs':30, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'lr_ae':1., # autoencoder learning rate
    'clip':5.0,  # gradient clipping, max norm        
    }
    return conf 

