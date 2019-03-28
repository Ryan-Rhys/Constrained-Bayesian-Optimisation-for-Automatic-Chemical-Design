"""
This module generates the latent features for the objective corresponding to
QED + SA + ring-penalty.
"""

import networkx as nx
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles
from rdkit.Chem import rdmolops

from Constrained_BO.Chemical_Design.ml.autoencoder.latent_space import encode_decode as lasp
from Constrained_BO.Chemical_Design.ml.qed import qed
import Constrained_BO.sascorer as sascorer

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

SA_scores = []
for i in range(len(smiles)):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))
    print(i)

cycle_scores = []
for i in range(len(smiles)):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[i]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)
    print(i)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
qed_values_normalized = (np.array(qed_values) - np.mean(qed_values)) / np.std(qed_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

smiles_one_hot_encoding = []
for i in range(len(smiles)):
    smiles_one_hot_encoding.append(preproc.smilelist_to_one_hot(smiles_rdkit[i]))
    print(i)

latent_points = []
for i in range(len(smiles_one_hot_encoding)):
    latent_points.append(encoder([smiles_one_hot_encoding[i]])[0][0])
    print(i)

# We store the results

latent_points = np.array(latent_points)
np.savetxt('qed_features_and_targets/latent_faetures.txt', latent_points)
targets = SA_scores_normalized + qed_values_normalized + cycle_scores_normalized
np.savetxt('qed_features_and_targets/targets.txt', targets)
np.savetxt('qed_features_and_targets/qed_values.txt', np.array(qed_values))
np.savetxt('qed_features_and_targets/SA_scores.txt', np.array(SA_scores))
np.savetxt('qed_features_and_targets/cycle_scores.txt', np.array(cycle_scores))
