from keras import backend as K
from keras.models import model_from_json
from keras.optimizers import Adam
from rdkit.Chem import MolFromSmiles, MolToSmiles
import numpy as np
import Constrained_BO.Chemical_Design.autoencoder.encoderdecoder.template_autoencoder as tvae
import h5py
import logging
import random
import os
os.environ["OMP_NUM_THREADS"] = '6'
os.environ["THEANO_FLAGS"] = 'floatX=float32,device=cpu,blas.ldflags="-lopenblas"'

DEF_LOSS = 'categorical_crossentropy'
DEF_OPT = Adam
DEF_LR = 0.001

LOG_FILENAME = 'encoderdecoder.log'

logging.getLogger('autoencoder')
logging.getLogger().setLevel(20)
# fileHandler = logging.FileHandler("{0}/{1}.log".format('./', LOG_FILENAME))
logging.getLogger().addHandler(logging.StreamHandler())


class PreProcessing(object):
    def __init__(self,
                 dataset,
                 max_len=120,
                 smiles=None):
        self.dataset = dataset
        self.chars = tvae.CONFIGS[dataset]['chars'] # 12 July 22:21 - This has the characters in it, not sure how to prevent decoding to small c.
        self.nchars = len(self.chars) # length is 36 for drugs dataset
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.max_len = max_len
        self.dataset_file = tvae.CONFIGS[dataset]['file']
        if type(smiles) is not list:
            smiles = [smiles]
        self.smiles = smiles

    def check_len_and_pad(self, string):
        if len(string) <= self.max_len:
            return string + " " * (self.max_len - len(string))

    def smilelist_to_one_hot(self, smiles_list):
        if type(smiles_list) is not list:
            if type(smiles_list) is str:
                smiles_list = smiles_list.split(',')
            else:
                raise TypeError("must pass a list of smiles")
        smiles_list = [self.check_len_and_pad(i) for i in smiles_list if self.check_len_and_pad(i)]
        Z = np.zeros((len(smiles_list),
                      self.max_len, self.nchars),
                      dtype=np.bool) # 249456*120*36
        for i, smile in enumerate(smiles_list):
            for t, char in enumerate(smile):
                Z[i, t, self.char_indices[char]] = 1
        return Z # big matrix with a 1 in it where the smiles character appears.

    def load_smile_set(self, path=None):
        if path:
            file = path
        else:
            file = self.dataset_file
        with open(file, 'r') as f:
            self.smiles = f.read().split

    def get_one_hot(self, smiles=None, total_smiles=100):
        if smiles:
            self.smiles = smiles
        if total_smiles == 'all':
            set_to_do = self.smiles
        else:
            set_to_do = random.sample(self.smiles, total_smiles)

        return self.smilelist_to_one_hot(set_to_do)


class EncoderDecoder(object):
    """An object to load and compile keras models.weight_file.
        It will also produce encoder and decoder functions and contain attributes about the nature of the model"""
    def __init__(self,
                 model_file='~/Documents/Machine_Learning/1_Project/Code/ml/autoencoder/champion_models/best_determ_model.json', # '~/a2g2_ml/ml/autoencoder/champion_models/best_determ_model.json'
                 weight_file='~/Documents/Machine_Learning/1_Project/Code/ml/autoencoder/champion_models/best_determ_weights.h5', # '~/a2g2_ml/ml/autoencoder/champion_models/best_determ_weights.h5'
                 model=None, regularizer_scale=None,
                 dataset='drugs', is_compiled=False,
                 preprocessor=None,
                 rnd_seed=None,
                 temperature=None,
                 output_sample=False):

        assert(not(model_file and model))
        self.encoder = None
        self.decoder = None
        self.is_compiled = is_compiled
        self.model_file = os.path.expanduser(model_file)
        self.model = model
        self.rnd_seed = rnd_seed
        self.temperature = temperature
        self.output_sample = output_sample
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = PreProcessing(dataset=dataset)
        if regularizer_scale is not None:
            self.regularizer_scale = regularizer_scale
        else:
            logging.info(' No regularizer_scale has been provided.'
                         ' If this is VAE, assuming the weights correspond to an annealed VAE'
                         ' and setting regularizer_scale = 1')
            self.regularizer_scale = 1
        if self.model_file and not self.model:
            self.make_model_from_file()
        else:
            raise StandardError("Need either a model or a model file")
        self.has_terminalrnn = (self.model.layers[-1].name == 'terminalgru')
        self.is_var = any([i.name == 'variationaldense' for i in self.model.layers])
        if self.is_var:
            logging.info('This is a variational autoencoder.')
            if self.output_sample:
                logging.info('Will output a sample from the distribution')
            else:
                logging.info('Will output the mean of the distribution')

        if not weight_file:
            pos_w_file = self.model_file.replace('model.json', 'weights.h5')
            if os.path.isfile(pos_w_file):
                logging.warning('No weight_file is given.'
                                ' Using weights from sibling file {}'.format(pos_w_file))
                weight_file = pos_w_file
        self.weight_file = os.path.expanduser(weight_file)

        if self.weight_file:
            self.load_weights_from_file()

    def make_model_from_file(self):
        if self.model_file:
            model_text = open(self.model_file, 'r').read()
        else:
            raise StandardError("You tried to create a model without inputing a file")

        if "variationaldense" in model_text:
            if "regularizer_scale" not in model_text:
                logging.info('Adding a regularizer_scale = {} to the VAE layer'.format(self.regularizer_scale))
                model_text = model_text.replace('"prior_logsigma"',
                                                '"regularizer_scale": {},\n "prior_logsigma"'.format(self.regularizer_scale))
            if "output_sample" not in model_text and self.output_sample:
                logging.info('Adding output_sample = {} to the VAE layer'.format(self.output_sample))
                model_text = model_text.replace('"prior_logsigma"',
                                                '"output_sample": true,\n "prior_logsigma"')

        if "terminalgru" in model_text:
            if self.rnd_seed:
                logging.info('Adding a rnd_seed parameter of {}'.format(self.rnd_seed))
                model_text = model_text.replace('"terminalgru",',
                                                '"terminalgru",\n"rnd_seed": {},'.format(self.rnd_seed))
            if self.temperature:
                logging.info('Adding a temperature parameter of {}'.format(self.temperature))
                model_text = model_text.replace('"terminalgru",',
                                                '"terminalgru",\n"temperature": {},'.format(self.temperature))

        self.model = model_from_json(model_text)

    def load_weights_from_file(self):
        with h5py.File(self.weight_file, mode='r') as f:
            for k in range(f.attrs['nb_layers']):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                w_shape = [i.shape for i in weights]
                logging.debug('Weights for this layer have shapes {}'.format(w_shape))
                try:
                    self.model.layers[k].set_weights(weights)
                except AssertionError:
                    logging.exception('Failed loading weights on layer {}. '
                                       'Weights initiated with random'.format(k))
                    continue

    def compile_model(self, loss=DEF_LOSS,
                      optimizer=DEF_OPT,
                      lr=DEF_LR):
        if self.is_compiled:
            logging.warning("Model is already compiled")
            return
        else:
            logging.warn("The model is not compiled, running compile_model"
                         "(loss, optimizer, lr) method for you"
                         " with pars {} {} {} ".format(DEF_LOSS, DEF_OPT, DEF_LR))

        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer

        if type(self.optimizer) is not str:
            optim = self.optimizer(lr=self.lr)
        else:
            optim = self.optimizer

        logging.info('Compiling model. This takes a while')
        self.model.compile(loss=self.loss,
                           optimizer=optim)
        self.is_compiled = True

    def get_model(self):
        return self.model

    def create_funtions(self):
        if not self.is_compiled:
            self.compile_model()
        if self.is_var:
            indices = [i for (i, layer) in enumerate(self.model.layers)
                       if layer.name == 'variationaldense']
        else:
            indices = [i for (i, layer) in enumerate(self.model.layers)
                       if layer.name == 'dense']
        index = np.int(np.mean(indices))
        if self.model.layers[index + 1].name == 'batchnormalization':
            index = index + 1
            logging.info('Middle layer followed by batchnorm. This layer ({})'
                         ' is chosen as latent space'.format(index))

        logging.info('Creating encoder and decoder function')
        self.encoder = K.Function([self.model.layers[0].get_input(train=False)],
                                  [self.model.layers[index].get_output(train=False)]) # index is the middle layer? i.e. the latent space?

        self.decoder = K.Function([self.model.layers[index + 1].get_input(train=False)],
                                  [self.model.layers[-1].get_output(train=False)],
                                  **{'on_unused_input': 'warn'})

    def get_functions(self):
        if not (self.encoder and self.decoder):
            self.create_funtions()
        return self.encoder, self.decoder

    def test_model_acc(self, test_set):
        if not self.is_compiled:
            self.compile_model()
        return self.model.test_on_batch(test_set, test_set,
                                        sample_weight=None, accuracy=True)


class PostProcessing(object):
    def __init__(self, encoderdecoder,
                 temperature=1.0,
                 max_len=120,
                 smiles=None,
                 dataset='drugs',
                 closest_or_mode='closest'):
        self.dataset = dataset
        self.chars = tvae.CONFIGS[dataset]['chars']
        self.nchars = len(self.chars)
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.max_len = max_len
        self.dataset_file = tvae.CONFIGS[dataset]['file']
        self.temperature = temperature
        self.encoder = self.decoder = None
        self.encoderdecoder = encoderdecoder
        self.encoder, self.decoder = self.encoderdecoder.get_functions()
        self.has_terminalrnn = self.encoderdecoder.has_terminalrnn
        assert((type(closest_or_mode) is str) and
               (closest_or_mode in ['mode', 'closest']))
        self.closest_or_mode = closest_or_mode

    def thermal_argmax(self, prob_arr):
        prob_arr = np.log(prob_arr) / self.temperature
        prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
        if np.greater_equal(prob_arr.sum(), 1.0000000001):
            logging.warn('Probabilities to sample add to more than 1, {}'.
                         format(prob_arr.sum()))
            prob_arr = prob_arr / (prob_arr.sum() + .0000000001)
        if np.greater_equal(prob_arr.sum(), 1.0000000001):
            logging.warn('Probabilities to sample still add to more than 1')
        return np.argmax(np.random.multinomial(1, prob_arr, 1))

    def one_hot_to_smiles(self, array):
        temp_string = ""
        for j in array:
            index = self.thermal_argmax(j)
            temp_string += self.indices_char[index]
        return temp_string

    def ls_to_smiles(self,
                     input_to_decode,
                     n_outputs,
                     max_attempts,
                     exclude_previous=True,
                     regularize_smiles=False,
                     calc_output_rep=True,
                     noise=0):
        all_smiles = []
        rdmols = []
        valid_smiles = []
        output_reps = []
        distances = []
        exclude = []

        def sort_results(rdmols, valid_smiles,
                         distances):
            sorted_rdmols = [x for (y, x) in sorted(zip(distances, rdmols))]
            sorted_valid_smiles = [x for (y, x) in sorted(zip(distances, valid_smiles))]
            sorted_distances = sorted(distances)
            return (sorted_rdmols, sorted_valid_smiles, sorted_distances)

        output = self.decoder(input_to_decode)

        counter = 0
        while len(rdmols) < n_outputs and counter < max_attempts:
            interim_input = [input_to_decode[0] + np.random.uniform(-noise,
                                                                    noise,
                                                                    size=len(input_to_decode))]
            counter += 1
            if self.has_terminalrnn is True:
                output = self.decoder(interim_input)
            else:
                logging.info("Last layer is not TGRU, will perform sampling outside Theano")
            smile = self.one_hot_to_smiles(output[0][0]).split(' ')[0]
            all_smiles.append(smile)

            if smile == '':
                continue
            if exclude_previous and any([smile == avoided_smile for avoided_smile in exclude]):
                continue

            mol = MolFromSmiles(smile)
            if mol is None:
                continue

            rdmols.append(mol)
            if regularize_smiles:
                smile = MolToSmiles(mol)
            valid_smiles.append(smile)
            exclude.append(smile)

            if calc_output_rep is True:
                one_hot_output = self.encoderdecoder.preprocessor.smilelist_to_one_hot([smile])
                output_rep = self.encoder([one_hot_output])[0]
                output_reps.append(output_rep)
                distances.append(np.linalg.norm(interim_input - output_rep))
        if calc_output_rep is True:
            sorted_outs = sort_results(rdmols, valid_smiles, distances)
            rdmols, valid_smiles, distances = sorted_outs

        return rdmols, valid_smiles, all_smiles, output_reps, distances

    def cs_to_ls_to_smiles(self,
                           one_hot_input,
                           end2end_attempts,
                           n_outputs,
                           decode_attempts=2):
        all_smiles = []
        rdmols = []
        valid_smiles = []
        output_reps = []
        distances = []

        if type(one_hot_input) is not list:
            one_hot_input = [one_hot_input]

        counter = 0
        while len(rdmols) < n_outputs and counter < end2end_attempts:
            counter += 1
            input_to_decode = self.encoder(one_hot_input)

            sampler_out = self.ls_to_smiles(input_to_decode, 1, decode_attempts)
            new_rdmols, new_valid_smiles, \
                new_all_smiles, new_output_reps, new_distances = sampler_out

            rdmols += new_rdmols
            valid_smiles += new_valid_smiles
            distances += new_distances
            all_smiles += new_all_smiles
            output_reps += new_output_reps
            distances += new_distances

        return rdmols, valid_smiles, all_smiles, output_reps, distances
        # closest_smile = get_closest_molecule(serious_smiles, distances)
        # super_distances.append(min(distances))
        # super_serious_smiles.append(closest_smile)
        # super_lista.append(Block(closest_smile).mol)
        # steps_taken.append(mixing_fraction)
        # print "Found {} molecules".format(len(lista))

