import numpy as np
import random
from ml.autoencoder.latent_space import encode_decode as lasp
import pickle
import os

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
from Constrained_BO.Chemical_Design.ml.autoencoder.latent_space.encode_decode import *
from Constrained_BO.Chemical_Design.ml.autoencoder.interpolators.interpolate import *
import copy
import time

limits = (0.0, 1.0)
n_steps = 10
use_random_mols = True
passes = 1
n_attempts = 2500
rnd_seed = None # random.randint(0, 1e6)
output_sample = False

TRAIN_SET = 'non_meth_oled_200k'

file = '/Users/rgbombarelli/a2g2_ml/ml/autoencoder/training_sets/non_methyl_oleds_200k.smi'

with open(file, 'r') as f:
    smiles = f.readlines()
smiles = [i.strip() for i in smiles]
preproc = lasp.PreProcessing(dataset=TRAIN_SET,
                             smiles=smiles)
preproc.load_smile_set(path=file)
test_set = preproc.get_one_hot(smiles, total_smiles=5000)

enc_dec = lasp.EncoderDecoder(model_file='~/a2g2_ml/ml/autoencoder/champion_models/best_oled_deter_model.json',
                              weight_file='~/a2g2_ml/ml/autoencoder/champion_models/best_oled_deter_weights.h5',
                              output_sample=output_sample,
                              rnd_seed=rnd_seed)

enc_dec.test_model_acc(test_set)

enc_dec_vae = lasp.EncoderDecoder(model_file='~/a2g2_ml/ml/autoencoder/champion_models/best_oled_vae_model.json',
                                  weight_file='~/a2g2_ml/ml/autoencoder/champion_models/best_oled_vae_weights.h5',
                                  output_sample=output_sample,
                                  rnd_seed=rnd_seed)

enc_dec_vae.test_model_acc(test_set)



TRAIN_SET = 'drugs'
file = '/Users/rgbombarelli/a2g2_ml/ml/autoencoder/training_sets/250k_rndm_zinc_drugs_clean.smi'
with open(file, 'r') as f:
    smiles = f.readlines()
smiles = [i.strip() for i in smiles]
preproc = lasp.PreProcessing(dataset=TRAIN_SET,
                             smiles=smiles)
preproc.load_smile_set(path=file)
test_set = preproc.get_one_hot(smiles, total_smiles=5000)


enc_dec_drug = lasp.EncoderDecoder(model_file='~/a2g2_ml/ml/autoencoder/champion_models/best_determ_model.json',
                              weight_file='~/a2g2_ml/ml/autoencoder/champion_models/best_determ_weights.h5',
                              output_sample=output_sample,
                              rnd_seed=rnd_seed)

enc_dec_drug.test_model_acc(test_set)

enc_dec_drug_vae = lasp.EncoderDecoder(model_file='~/a2g2_ml/ml/autoencoder/champion_models/best_vae_model.json',
                                  weight_file='~/a2g2_ml/ml/autoencoder/champion_models/best_vae_annealed_weights.h5',
                                  output_sample=output_sample,
                                  rnd_seed=rnd_seed)

enc_dec_drug_vae.test_model_acc(test_set)
