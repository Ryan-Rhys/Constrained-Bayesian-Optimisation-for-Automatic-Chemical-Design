"""
This module provides an constrained Bayesian Optimisation implementation for
generating molecules that are optimised for QED alone.
"""

import sys

import numpy as np
import scipy.stats as sps
from Constrained_BO.Chemical_Design.autoencoder.latent_space import encode_decode as lasp
from Constrained_BO.Chemical_Design.qed import qed
from rdkit.Chem import MolFromSmiles

from Constrained_BO.black_box_alpha import BB_alpha
from sparse_gp import SparseGP
from Constrained_BO.utils import load_object, save_object, load_data

np.random.seed(1)

# We load the data

# Classification data

X_tr_pos_con = load_object('../train_test_sets/Train_Samples/Positive_Latents/X_con_tr_pos20.dat')
X_tr_neg_con = load_object('../train_test_sets/Train_Samples/Negative_Latents/X_con_tr_neg20.dat')
y_tr_pos_con = load_object('../train_test_sets/Train_Samples/Positive_Latents/y_con_tr_pos20.dat')
y_tr_neg_con = load_object('../train_test_sets/Train_Samples/Negative_Latents/y_con_tr_neg20.dat')

# Balance the number of samples from each class 

m = X_tr_pos_con.shape[0]
permute_pos = np.random.choice(m, m, replace=False)
n = X_tr_neg_con.shape[0]
permute_neg = np.random.choice(n, n, replace=False)

X_tr_pos_con = X_tr_pos_con[permute_pos, :]
X_tr_neg_con = X_tr_neg_con[permute_neg, :]
y_tr_pos_con = y_tr_pos_con[permute_pos]
y_tr_neg_con = y_tr_neg_con[permute_neg]

if m >= n:
    o = n
else:
    o = m

# o is the size of the smaller dataset

X_tr_pos_con = X_tr_pos_con[0:o]
X_tr_neg_con = X_tr_neg_con[0:o]
y_tr_pos_con = y_tr_pos_con[0:o]
y_tr_neg_con = y_tr_neg_con[0:o]

# concatenate the positive and negative labels

X_tr_con = np.concatenate((X_tr_pos_con, X_tr_neg_con))
y_tr_con = np.concatenate((y_tr_pos_con, y_tr_neg_con))

# shuffle the data (important as the first data set has all the positive labels first)

n = X_tr_con.shape[0]
permutation_con = np.random.choice(n, n, replace=False)

# train/test sets slightly unbalanced in terms of labels due to random partitioning

X_train_con = X_tr_con[permutation_con, :][0: np.int(np.round(0.9 * n)), :]  # 90/10 train/test split
X_test_con = X_tr_con[permutation_con, :][np.int(np.round(0.9 * n)):, :]

# typecasting to ints so that num_labels can be computed later on

y_train_con = np.int64(y_tr_con[permutation_con][0: np.int(np.round(0.9 * n))])
y_test_con = np.int64(y_tr_con[permutation_con][np.int(np.round(0.9 * n)):])

# Regression Data

X = np.loadtxt('../solo_qed_features_and_targets/latent_faetures.txt')
y = -np.loadtxt('../solo_qed_features_and_targets/targets.txt')
y = y.reshape((-1, 1)) 

n = X.shape[0]
permutation = np.random.choice(n, n, replace=False)

X_train = X[ permutation, :][0: np.int(np.round(0.9 * n)), :]  # 90/10 train/test split.
X_test = X[ permutation, :][np.int(np.round(0.9 * n)):, :]

y_train = y[permutation][0: np.int(np.round(0.9 * n))]
y_test = y[permutation][np.int(np.round(0.9 * n)):]

# We set the number of iterations for data collection

num_iters = 20

for iteration in range(num_iters):

    # We fit the GP

    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,
                       y_test, minibatch_size=10 * M, max_iterations=45, learning_rate=0.005)
    save_object(sgp, "results_QED_solo/sgp{}.dat".format(iteration))

    # We load the saved gp

    sgp = load_object("results_QED_solo/sgp{}.dat".format(iteration))

    # We load some previous trained gp

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale=np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)
    
    # We train the classification model
    
    # We load the random seed

    seed = 1
    np.random.seed(seed)

    # We load the data
        
    datasets, n, d, n_labels = load_data(X_train_con, y_train_con, X_test_con, y_test_con)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
        
    N_train = train_set_x.get_value(borrow=True).shape[0]
    N_test = test_set_x.get_value(borrow=True).shape[0]
    layer_sizes = [d, 100, 100, n_labels]
    n_samples = 50
    alpha = 0.5
    learning_rate = 0.001
    v_prior = 1.0
    batch_size = 5000
    print('... building model')
    sys.stdout.flush()
    bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size,
                        train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test)
    print('... training')
    sys.stdout.flush()
    
    test_error, test_ll = bb_alpha.train(500) # 2 August - 10 epochs to train small dataset for 20%

    # 4 September - saving the BNN is optional
    
    # We save the trained BNN
    
    #sys.setrecursionlimit(4000)
    
    # don't have to save the BNN because it won't be necessary to plot the predictions in 56-D space.
    
    # save_object(bb_alpha, "results_QED_solo/bb_alpha{}.dat".format(iteration))

    # We pick the next 50 inputs (4 September - variable name bb_alpha_samples not descriptive, it is the size of the batch of collected data points)
    
    bb_alpha_samples = 50

    next_inputs = sgp.batched_greedy_ei(bb_alpha, 50, np.min(X_train, 0), np.max(X_train, 0), bb_alpha_samples)
    
    # We load the decoder to obtain the molecules

    preproc = lasp.PreProcessing(dataset='drugs')
    enc_dec = lasp.EncoderDecoder()
    encoder, decoder = enc_dec.get_functions()

    postprocessor = lasp.PostProcessing(enc_dec)
    
    # We collect the molecule statistics
    
    # 4 September - need descriptions of these variables, highly unclear
    
    # 4 September - variables are lists of length = decode_attempts
    
    num_C = []  # number of molecules decoded to methane ('C' in SMILES)
    num_val = []  # number of molecules decoded to valid structures
    num_sensibles = []  # number of molecules decoded to valid structures AND NOT methane
    num_sensible_and_longs = []  # number of molecules decoded to valid structures AND NOT methane AND length > 5

    valid_smiles_final = []  # list of molecules decoded to valid structures AND NOT methane AND length > 5
    all_smiles_final = []  # list of all decoded molecules
    
    decode_attempts = 100
        
    for i in range(next_inputs.shape[0]):
        
        # decode the collected latent data points to SMILES strings
        
        sampler_out = postprocessor.ls_to_smiles([next_inputs[i: (i + 1), :]], decode_attempts, decode_attempts,)
        rdmols, valid_smiles, all_smiles, output_reps, distances = sampler_out
        
        num_methanes = all_smiles.count('C')
        num_valid_mols = sum([all_smiles.count(x) for x in valid_smiles])
        num_sensible = num_valid_mols - num_methanes
        valid_sens_smiles = [x for x in valid_smiles if len(x) > 5]
        num_sensible_and_long = sum([all_smiles.count(x) for x in valid_sens_smiles])
                
        num_C.append(num_methanes)
        num_val.append(num_valid_mols)
        num_sensibles.append(num_sensible)
        num_sensible_and_longs.append(num_sensible_and_long)
        
        valid_smiles_final.append(valid_sens_smiles)
        all_smiles_final.append(all_smiles)
        
        print(i)
        
    save_object(num_C, "results_QED_solo/num_C_samples{}.dat".format(iteration))
    save_object(num_val, "results_QED_solo/num_val_samples{}.dat".format(iteration))
    
    save_object(num_sensibles, "results_QED_solo/num_sensible{}.dat".format(iteration))
    save_object(num_sensible_and_longs, "results_QED_solo/num_sensible_and_long{}.dat".format(iteration))

    valid_smiles_final_final = []
    new_features = []
    
    # We collect the constraint labels and probabilities
    
    con_scores = np.int64(np.zeros([next_inputs.shape[0], ]))  # collect labels
    probs = []  # collect the probabilities of satisfying the constraint
    log_probs = []  # collect the log probabilities of satisfying the constraint
        
    for i in range(len(valid_smiles_final)):
        
        probs.append(bb_alpha.prediction_probs(next_inputs[i].reshape(1, d))[0][0][1])
        log_probs.append(bb_alpha.pred_log_probs(next_inputs[i].reshape(1, d))[0][0][1])
        
        if num_sensible_and_longs[i] > 20:
            
            valid_smiles_final_final.append([valid_smiles_final[i][0]])
            new_features.append(next_inputs[i, :])
            con_scores[i] = 1
            
        else:
            con_scores[i] = 0
                        
    new_features = np.array(new_features)
    valid_smiles_final = valid_smiles_final_final

    save_object(valid_smiles_final, "results_QED_solo/valid_smiles{}.dat".format(iteration))
    save_object(all_smiles_final, "results_QED_solo/all_smiles{}.dat".format(iteration))
    save_object(con_scores, "results_QED_solo/con_scores{}.dat".format(iteration))
    save_object(probs, "results_QED_solo/probs{}.dat".format(iteration))
    save_object(log_probs, "results_QED_solo/log_probs{}.dat".format(iteration))

    qed_values = np.loadtxt('../solo_qed_features_and_targets/qed_values.txt')
    qed_values_normalized = (np.array(qed_values) - np.mean(qed_values)) / np.std(qed_values)

    targets = qed_values_normalized

    reg_scores = []  # collect scores for objective function
    qed_scores = []  # collect scores for qed term in objective function
        
    for i in range(len(valid_smiles_final)):
        to_add = []
        qed_store = []
        if len(valid_smiles_final[i]) != 0:
            for j in range(0, len(valid_smiles_final[ i ])):
                current_qed_value = qed.default(MolFromSmiles(valid_smiles_final[i][j]))
                current_qed_value_normalized = (current_qed_value - np.mean(qed_values)) / np.std(qed_values)
                score = current_qed_value_normalized
                to_add.append(-score)
                qed_store.append(current_qed_value)
        reg_scores.append(to_add)
        qed_scores.append(qed_store)
        print(i)

    print(valid_smiles_final)
    print(reg_scores)

    save_object(reg_scores, "results_QED_solo/reg_scores{}.dat".format(iteration))
    save_object(qed_scores, "results_QED_solo/qed_scores{}.dat".format(iteration))
        
    if new_features.shape != (0,):

        X_train = np.concatenate([X_train, new_features], 0)
        y_train = np.concatenate([y_train, np.array(reg_scores)], 0)
        
    X_train_con = np.concatenate([X_train_con, next_inputs], 0)
    y_train_con = np.concatenate([y_train_con, con_scores], 0)
