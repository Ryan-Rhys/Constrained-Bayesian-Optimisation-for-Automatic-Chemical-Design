"""
This module generates the latent features for the objective corresponding to
QED alone.
"""

import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles

from Constrained_BO.Chemical_Design.autoencoder.latent_space import encode_decode as lasp
from Constrained_BO.Chemical_Design.qed import qed

# We load the smiles data

fname = '../training_sets/250k_rndm_zinc_drugs_clean.smi'

with open(fname) as f:
    smiles = f.readlines()

for i in range(len(smiles)):
    smiles[i] = smiles[i].strip()

# We load the auto-encoder

preproc = lasp.PreProcessing(dataset='drugs')
enc_dec = lasp.EncoderDecoder()
encoder, decoder = enc_dec.get_functions()

smiles_rdkit = []
for i in range(len(smiles)):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[i])))
    print(i)

qed_values = []
for i in range(len(smiles)):
    qed_values.append(qed.default(MolFromSmiles(smiles_rdkit[i])))
    print(i)

qed_values_normalized = (np.array(qed_values) - np.mean(qed_values)) / np.std(qed_values)

smiles_one_hot_encoding = []
for i in range(len(smiles)):
    smiles_one_hot_encoding.append(preproc.smilelist_to_one_hot(smiles_rdkit[i]))
    print(i)

latent_points= []
for i in range(len(smiles_one_hot_encoding)):
    latent_points.append(encoder([smiles_one_hot_encoding[i]])[0][0])
    print(i)

# We store the results

latent_points = np.array(latent_points)
np.savetxt('solo_qed_features_and_targets/latent_faetures.txt', latent_points)
targets = qed_values_normalized
np.savetxt('solo_qed_features_and_targets/targets.txt', targets)
np.savetxt('solo_qed_features_and_targets/qed_values.txt', np.array(qed_values))
