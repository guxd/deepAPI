def config_AHRED():
    conf = {
    'maxlen':30, # maximum sequence length

# Model Arguments
    'emb_size':200, # size of word embeddings
    'n_hidden':300, # number of hidden units per layer
    'n_layers':1, # number of layers
    'noise_radius':0.2, # stdev of noise for autoencoder (regularizer)
    'noise_anneal':0.995, # anneal noise_radius exponentially by this every 100 iterations
    'arch_g':'100-100', # prior generator architecture (MLP)
    'arch_q':'100-100-100', # posterior generator architecture (MLP)
    'arch_d':'600-600', # critic/discriminator architecture (MLP)
    'e_size':100, # dimension of the gaussian noise
    'z_size':100, # dimension of z # 300 performs worse
    'lambda_gp':10, # Gradient penalty lambda hyperparameter.
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'enc_grad_norm':True, # norm code gradient from critic->encoder
    'gan_to_enc':-0.01, # weight factor passing gradient from gan to encoder
    'dropout':0.5, # dropout applied to layers (0 = no dropout)

# Training Arguments
    'batch_size':32,
    'epochs':20, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'n_iters_d':5, # number of discriminator iterations in training
    'lr_ae':1.0, # autoencoder learning rate
    'lr_gan_g':5e-05, # generator learning rate
    'lr_gan_d':1e-05, # critic/discriminator learning rate
    'beta1':0.9, # beta1 for adam
    'clip':1.0,  # gradient clipping, max norm
    'gan_clamp':0.01,  # WGAN clamp (Do not use clamp when you apply gradient penelty             
    }
    return conf 

def config_AHRED_GMP():
    conf=config_AHRED()
    conf['n_prior_components']=3  # DailyDial 3 SWDA 3
    conf['gumbel_temp']=0.1
    return conf

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

    'n_iters_d':0, # number of discriminator iterations in training
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

    'n_iters_d':0, # number of discriminator iterations in training
    'lr_ae':1., # autoencoder learning rate
    'clip':5.0,  # gradient clipping, max norm        
    }
    return conf 

