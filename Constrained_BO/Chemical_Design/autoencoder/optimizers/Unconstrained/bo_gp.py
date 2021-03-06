"""
This module provides a Bayesian Optimisation implementation for generating 
molecules that are optimised for logP, SA and don't incur a ring-penalty.
"""

import networkx as nx
import numpy as np
import scipy.stats as sps
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops

from Constrained_BO.Chemical_Design.autoencoder.latent_space import encode_decode as lasp
import Constrained_BO.sascorer as sascorer
from sparse_gp import SparseGP
from Constrained_BO.utils import save_object, load_object


np.random.seed(1)

# We load the data

# Regression Data

X = np.loadtxt('../latent_features_and_targets/latent_faetures.txt')
y = -np.loadtxt('../latent_features_and_targets/targets.txt')
y = y.reshape((-1, 1))

n = X.shape[0]
permutation = np.random.choice(n, n, replace=False)

X_train = X[permutation, :][0: np.int(np.round(0.9 * n)), :]  # 90/10 train/test split.
X_test = X[permutation, :][np.int(np.round(0.9 * n)):, :]

y_train = y[permutation][0: np.int(np.round(0.9 * n))]
y_test = y[permutation][np.int(np.round(0.9 * n)):]

num_iters = 20

for iteration in range(num_iters):

    # We fit the GP

    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,
                       y_test, minibatch_size=10 * M, max_iterations=50, learning_rate=0.005)
    save_object(sgp, "results_logP/sgp{}.dat".format(iteration))

    # We load the saved gp

    sgp = load_object("results_logP/sgp{}.dat".format(iteration))

    # We load some previous trained gp

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))
    
    # We load the decoder to obtain the molecules

    preproc = lasp.PreProcessing(dataset='drugs')
    enc_dec = lasp.EncoderDecoder()
    encoder, decoder = enc_dec.get_functions()

    postprocessor = lasp.PostProcessing(enc_dec)
    
    # We collect the molecule statistics
    
    num_C = []
    num_val = []
    num_sensibles = []
    num_sensible_and_longs = []

    valid_smiles_final = []
    valid_sens_smiles_final = []
    all_smiles_final = []
    
    decode_attempts = 100
        
    for i in range(next_inputs.shape[0]):
        
        sampler_out = postprocessor.ls_to_smiles([next_inputs[i : (i + 1), :]], decode_attempts, decode_attempts,)
        rdmols, valid_smiles, all_smiles, output_reps, distances = sampler_out
                
        valid_smiles_final.append(valid_smiles)
        
        num_methanes = all_smiles.count('C')
        num_valid_mols = sum([all_smiles.count(x) for x in valid_smiles])
        num_sensible = num_valid_mols - num_methanes
        valid_sens_smiles = [x for x in valid_smiles if len(x) > 5]
        num_sensible_and_long = sum([all_smiles.count(x) for x in valid_sens_smiles])
                
        num_C.append(num_methanes)
        num_val.append(num_valid_mols)
        num_sensibles.append(num_sensible)
        num_sensible_and_longs.append(num_sensible_and_long)
        
        valid_sens_smiles_final.append(valid_sens_smiles)
        all_smiles_final.append(all_smiles)
        
        print(i)
        
    save_object(num_C, "results_logP/num_C_samples{}.dat".format(iteration))
    save_object(num_val, "results_logP/num_val_samples{}.dat".format(iteration))
    
    save_object(num_sensibles, "results_logP/num_sensible{}.dat".format(iteration))
    save_object(num_sensible_and_longs, "results_logP/num_sensible_and_long{}.dat".format(iteration))
    
    save_object(all_smiles_final, "results_logP/all_smiles{}.dat".format(iteration))

    valid_smiles_final_final = []
    valid_sens_smiles_final_final = []
    new_features = []
    
    for i in range(len(valid_smiles_final)):
        
        if len(valid_smiles_final[i]) > 0:
            valid_smiles_final_final.append([ valid_smiles_final[i][0]])
            new_features.append(next_inputs[i, :])
            
    new_features = np.array(new_features)  # 5 August: adding features if they're valid points, I don't want to introduce an artificial constraint.
    valid_smiles_final = valid_smiles_final_final
            
    for i in range(len(valid_sens_smiles_final)):
        
        if num_sensible_and_longs[i] > 20:
            
            valid_sens_smiles_final_final.append([ valid_sens_smiles_final[i][0]])
            
    valid_sens_smiles_final = valid_sens_smiles_final_final
    save_object(valid_smiles_final, "results_logP/valid_smiles{}.dat".format(iteration))
    save_object(valid_sens_smiles_final, "results_logP/valid_sens_smiles{}.dat".format(iteration))

    logP_values = np.loadtxt('../latent_features_and_targets/logP_values.txt')
    SA_scores = np.loadtxt('../latent_features_and_targets/SA_scores.txt')
    cycle_scores = np.loadtxt('../latent_features_and_targets/cycle_scores.txt')
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized

    reg_scores = []  # collect y-values for objective function
    logP_scores = []
    SA_values = []
    
    for i in range(len(valid_smiles_final)):
        to_add = []
        logP = []
        SA = []
        if len(valid_smiles_final[i]) != 0:
            for j in range(0, len(valid_smiles_final[i])):
                current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[i][j]))
                current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[i][j]))
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[i][j]))))

                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([len(j) for j in cycle_list])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
          
                current_cycle_score = -cycle_length

                current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
                current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
                current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

                score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
                to_add.append(-score)
                logP.append(current_log_P_value)
                SA.append(current_SA_score)
                
        reg_scores.append(to_add)
        logP_scores.append(logP)
        SA_values.append(SA)
        print(i)
        
    print(valid_smiles_final)
    print(reg_scores)

    save_object(reg_scores, "results_logP/reg_scores{}.dat".format(iteration))
    save_object(logP_scores, "results_logP/logP_scores{}.dat".format(iteration))
    save_object(SA_values, "results_logP/SA_scores{}.dat".format(iteration))
    
    if new_features.shape != (0,):

        X_train = np.concatenate([X_train, new_features], 0)
        y_train = np.concatenate([y_train, np.array(reg_scores)], 0)
