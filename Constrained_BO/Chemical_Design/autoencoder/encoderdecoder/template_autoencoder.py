import warnings
import numpy as np
from random import shuffle
import random
import time
import sys
import os
sys.path
sys.path.append(os.path.expanduser("~/seya"))
sys.path.append(os.path.expanduser("~/keras"))

from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector
from keras.layers.core import TimeDistributedDense, TimeDistributedMerge, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, TerminalGRU
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, VAEWeightAnnealer
from keras.layers.variational import VariationalDense as VAE

MAX_LEN = 120
TRAIN_SET = 'drugs'
TEMP = np.array(1.00, dtype=np.float32)
PADDING = 'right'
VAL_SPLIT = 0.1
DOUBLE_HG = True
LOSS = 'categorical_crossentropy'
REPEAT_VECTOR = True
VAE_WEIGHTS_START = 4

ACTIVATION = random.choice(['tanh'])
AVERAGE_POOL = random.choice([True, False])
BATCHNORM_CONV = random.choice([True, False])
BATCHNORM_GRU = random.choice([True, False])
BATCHNORM_MID = random.choice([True, False])
BATCHNORM_VAE = random.choice([True, False])
BATCH_SIZE = int(10**random.uniform(1.7, 2.3))
CONV_ACTIVATION = random.choice(['tanh'])
CONV_DEPTH = random.randint(3, 8)
CONV_DIM_DEPTH = int(2**random.uniform(1, 16))
CONV_DIM_WIDTH = int(2**random.uniform(1, 16))
CONV_D_GROWTH_FACTOR = random.uniform(0.5, 2)
CONV_W_GROWTH_FACTOR = random.uniform(0.5, 2)
DO_CONV_ENCODER = random.choice([True, False])
DO_EXTRA_GRU = random.choice([True, False])
DO_VAE = random.choice([True, False])
EPOCHS = int(10**random.uniform(2.0, 3.0))
GRU_DEPTH = random.randint(2, 5)
HG_GROWTH_FACTOR = random.uniform(.5, 2)
HIDDEN_DIM = int(10**random.uniform(1.7, 2.7))
LR = 10**random.uniform(-3.6, -2.6)
MIDDLE_LAYER = random.randint(1, 6)
MOMENTUM = random.uniform(.85, .999)
OPTIM = random.choice(['adam'])
RECURRENT_DIM = int(10**random.uniform(1.7, 3.0))
RNN_ACTIVATION = random.choice(['tanh'])
TGRU_DROPOUT = random.uniform(.00, .25)
VAE_ACTIVATION = random.choice(['tanh'])
VAE_SIGMOID_SLOPE = random.uniform(0.5, 1.0)
VAE_ANNEALER_START = random.randint(EPOCHS / 20, EPOCHS / 2)

DEF_PRMS = {'activation': ACTIVATION, 'average_pooling': AVERAGE_POOL,
                  'batchnorm_conv': BATCHNORM_CONV,
                  'batchnorm_gru': BATCHNORM_GRU, 'batchnorm_mid': BATCHNORM_MID,
                  'batch_size': BATCH_SIZE, 'conv_activation': CONV_ACTIVATION,
                  'conv_depth': CONV_DEPTH, 'conv_dim_depth': CONV_DIM_DEPTH,
                  'conv_dim_width': CONV_DIM_WIDTH,
                  'conv_d_growth_factor': CONV_D_GROWTH_FACTOR,
                  'conv_w_growth_factor': CONV_W_GROWTH_FACTOR, 'do_extra_gru': DO_EXTRA_GRU,
                  'do_vae': DO_VAE, 'do_conv_encoder': DO_CONV_ENCODER,
                  'epochs': EPOCHS,
                  'gru_depth': GRU_DEPTH, 'hg_growth_factor': HG_GROWTH_FACTOR,
                  'hidden_dim': HIDDEN_DIM, 'loss': LOSS, 'lr': LR,
                  'middle_layer': MIDDLE_LAYER,
                  'momentum': MOMENTUM, 'optim': OPTIM,
                  'rnn_activation': RNN_ACTIVATION,
                  'vae_annealer_start': VAE_ANNEALER_START,
                  'tgru_dropout': TGRU_DROPOUT,
                  'batchnorm_vae': BATCHNORM_VAE,
                  'vae_activation': VAE_ACTIVATION, 'vae_sigmoid_slope': VAE_SIGMOID_SLOPE,
                  'set': TRAIN_SET, 'recurrent_dim': RECURRENT_DIM}

CONFIGS = {
    'mini_blas': {'file': '/n/aagfs01/samsung/notebooks/Rafa/all_blas_mini_smiles.smi',
                  'chars': [u' ', u'#', u'%', u'(', u')', u'-', u'0', u'1', u'2', u'3',
                           u'4', u'5', u'6', u'7', u'8', u'9', u'=', u'C', u'F', u'H',
                           u'N', u'O', u'S', u'[', u']', u'c', u'n', u'o', u's']},
   'blas': {'file': '/n/aagfs01/samsung/notebooks/Rafa/all_blas_smiles.smi',
            'chars': [u' ', u'#', u'%', u'(', u')', u'-', u'0', u'1', u'2', u'3', u'4',
                      u'5', u'6', u'7', u'8', u'9', u'=', u'C', u'F', u'H', u'N', u'O',
                      u'S', u'[', u']', u'c', u'n', u'o', u's']},
   'all': {'file': '/n/aagfs01/samsung/notebooks/Rafa/all_smiles.smi',
           'chars': [u' ', u'#', u'%', u'(', u')', u'+', u'-', u'0', u'1', u'2', u'3', u'4',
                     u'5', u'6', u'7', u'8', u'9', u'=', u'B', u'C', u'F', u'G', u'H', u'N',
                     u'O', u'P', u'S', u'[', u']', u'c', u'e', u'i', u'n', u'o', u'r', u's']},
    'drugs': {'file': '/n/aagfs01/samsung/notebooks/Rafa/250k_rndm_zinc_drugs_clean.smi',
              'chars': [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
                        '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
                        'c', 'l', 'n', 'o', 'r', 's']},
   'non_meth_oled': {'file': '/n/aagfs01/samsung/notebooks/Rafa/non_methyl_oleds.smi',
                     'chars': [u' ', u'#', u'%', u'(', u')', u'+', u'-', u'0', u'1', u'2', u'3',
                               u'4', u'5', u'6', u'7', u'8', u'9', u'=', u'B', u'C', u'F', u'G',
                               u'H', u'N', u'O', u'P', u'S', u'[', u']', u'c', u'e', u'i', u'n',
                               u'o', u'r', u's']},
   'non_meth_oled_200k': {'file': '/n/aagfs01/samsung/notebooks/Rafa/non_methyl_oleds_200k.smi',
                          'chars': [u' ', u'#', u'%', u'(', u')', u'+', u'-', u'0', u'1', u'2',
                                    u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'=', u'B', u'C',
                                    u'F', u'G', u'H', u'N', u'O', u'P', u'S', u'[', u']', u'c',
                                    u'e', u'i', u'n', u'o', u'r', u's']}}

chars = CONFIGS[TRAIN_SET]['chars']
nchars = len(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def smile_convert(string):
    if len(string) < MAX_LEN:
        if PADDING == 'right':
            return string + " " * (MAX_LEN - len(string))
        elif PADDING == 'left':
            return " " * (MAX_LEN - len(string)) + string
        elif PADDING == 'none':
            return string


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class CheckMolecule(Callback):
    def on_epoch_end(self, epoch, logs={}):
        test_smiles = ["c1ccccc1"]
        test_smiles = [smile_convert(i) for i in test_smiles]
        Z = np.zeros((len(test_smiles), MAX_LEN, nchars), dtype=np.bool)
        for i, smile in enumerate(test_smiles):
            for t, char in enumerate(smile):
                Z[i, t, char_indices[char]] = 1

        string = ""
        for i in self.model.predict(Z):
            for j in i:
                index = sample(j, TEMP)
                string += indices_char[index]
#        print "\n" + string # 19 July


class CheckpointPostAnnealing(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', start_epoch=0):
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose,
                                 save_best_only=save_best_only, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.start_epoch:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                self.model.save_weights(filepath, overwrite=True)


def main(parameters=DEF_PRMS,
         weight_file='weights.h5',
         model_file='model.json'):
    for key in DEF_PRMS:
        if key not in parameters:
            parameters[key] = DEF_PRMS[key]
        if type(parameters[key]) in [float, np.ndarray]:
            parameters[key] = np.float(parameters[key])
#        print key, parameters[key] # 19 July

    def no_schedule(x):
        return float(1)

    def sigmoid_schedule(x, slope=1., start=parameters['vae_annealer_start']):
        return float(1 / (1. + np.exp(slope * (start - float(x)))))

    start = time.time()
    with open(CONFIGS[TRAIN_SET]['file'], 'r') as f:
        smiles = f.readlines()

    smiles = [i.strip() for i in smiles]
    print 'Training set size is', len(smiles)
    smiles = [smile_convert(i) for i in smiles if smile_convert(i)]
    print 'Training set size is {}, after filtering to max length of {}'.format(len(smiles), MAX_LEN)
    shuffle(smiles)

    #print('total chars:', nchars) # 19 July

    X = np.zeros((len(smiles), MAX_LEN, nchars), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            X[i, t, char_indices[char]] = 1

    model = Sequential()

    ## Convolutions
    if parameters['do_conv_encoder']:
        model.add(Convolution1D(int(parameters['conv_dim_depth'] *
                                    parameters['conv_d_growth_factor']),
                                int(parameters['conv_dim_width'] *
                                    parameters['conv_w_growth_factor']),
                                batch_input_shape=(parameters['batch_size'], MAX_LEN, nchars),
                                activation=parameters['conv_activation']))

        if parameters['batchnorm_conv']:
            model.add(BatchNormalization(mode=0, axis=-1))
        if parameters['average_pooling']:
            model.add(AveragePooling1D())

        for j in range(parameters['conv_depth'] - 1):
            model.add(Convolution1D(int(parameters['conv_dim_depth'] *
                                        parameters['conv_d_growth_factor']**(j + 1)),
                                    int(parameters['conv_dim_width'] *
                                        parameters['conv_w_growth_factor']**(j + 1)),
                                    activation=parameters['conv_activation']))
            if parameters['batchnorm_conv']:
                model.add(BatchNormalization(mode=0, axis=-1))
            if parameters['average_pooling']:
                model.add(AveragePooling1D())

        if parameters['do_extra_gru']:
            model.add(GRU(parameters['recurrent_dim'],
                      return_sequences=False,
                      activation=parameters['rnn_activation']))
        else:
            model.add(Flatten())

    else:
        for k in range(parameters['gru_depth'] - 1):
            model.add(GRU(parameters['recurrent_dim'], return_sequences=True,
                          batch_input_shape=(parameters['batch_size'], MAX_LEN, nchars),
                          activation=parameters['rnn_activation']))
            if parameters['batchnorm_gru']:
                model.add(BatchNormalization(mode=0, axis=-1))

        model.add(GRU(parameters['recurrent_dim'],
                      return_sequences=False,
                      activation=parameters['rnn_activation']))
        if parameters['batchnorm_gru']:
            model.add(BatchNormalization(mode=0, axis=-1))

    ## Middle layers
    for i in range(parameters['middle_layer']):
        model.add(Dense(int(parameters['hidden_dim'] *
                            parameters['hg_growth_factor']**(parameters['middle_layer'] - i)),
                        activation=parameters['activation']))
        if parameters['batchnorm_mid']:
            model.add(BatchNormalization(mode=0, axis=-1))

    ## Variational AE
    if parameters['do_vae']:
        model.add(VAE(parameters['hidden_dim'], batch_size=parameters['batch_size'],
                      activation=parameters['vae_activation'],
                      prior_logsigma=0))
        if parameters['batchnorm_vae']:
            model.add(BatchNormalization(mode=0, axis=-1))

    if DOUBLE_HG:
        for i in range(parameters['middle_layer']):
            model.add(Dense(int(parameters['hidden_dim'] *
                                parameters['hg_growth_factor']**(i)),
                            activation=parameters['activation']))
            if parameters['batchnorm_mid']:
                model.add(BatchNormalization(mode=0, axis=-1))

    if REPEAT_VECTOR:
        model.add(RepeatVector(MAX_LEN))

    ## Recurrent for writeout
    for k in range(parameters['gru_depth'] - 1):
        model.add(GRU(parameters['recurrent_dim'], return_sequences=True,
                      activation=parameters['rnn_activation']))
        if parameters['batchnorm_gru']:
            model.add(BatchNormalization(mode=0, axis=-1))

    model.add(TerminalGRU(nchars, return_sequences=True,
                          activation='softmax',
                          temperature=TEMP,
                          dropout_U=parameters['tgru_dropout']))

    if OPTIM == 'adam':
        optim = Adam(lr=parameters['lr'], beta_1=parameters['momentum'])
    elif OPTIM == 'rmsprop':
        optim = RMSprop(lr=parameters['lr'], beta_1=parameters['momentum'])
    elif OPTIM == 'sgd':
        optim = SGD(lr=parameters['lr'], beta_1=parameters['momentum'])

    model.compile(loss=LOSS, optimizer=optim)

    # SAVE

    json_string = model.to_json()
    open(model_file, 'w').write(json_string)

   # print parameters # 19 July

    # CALLBACK
    smile_checker = CheckMolecule()

    cbk = ModelCheckpoint(weight_file,
                          save_best_only=True)

    if parameters['do_vae']:
        for i, layer in enumerate(model.layers):
            if layer.name == 'variationaldense':
                vae_index = i

        vae_schedule = VAEWeightAnnealer(sigmoid_schedule,
                                         vae_index,
                                         )
        anneal_epoch = parameters['vae_annealer_start']
        weights_start = anneal_epoch + int(min(VAE_WEIGHTS_START, 0.25 * anneal_epoch))

        cbk_post_VAE = CheckpointPostAnnealing('annealed_' + weight_file,
                                               save_best_only=True,
                                               monitor='val_acc',
                                               start_epoch=weights_start,
                                               verbose=1)

        model.fit(X, X, batch_size=parameters['batch_size'],
                  nb_epoch=parameters['epochs'],
                  callbacks=[smile_checker, vae_schedule, cbk, cbk_post_VAE],
                  validation_split=VAL_SPLIT,
                  show_accuracy=True)
    else:
        model.fit(X, X, batch_size=parameters['batch_size'],
                  nb_epoch=parameters['epochs'],
                  callbacks=[smile_checker, cbk],
                  validation_split=VAL_SPLIT,
                  show_accuracy=True)

    end = time.time()
    #print parameters # 19 July
    #print(end - start), 'seconds elapsed' # 19 July


if __name__ == "__main__":
    main()
