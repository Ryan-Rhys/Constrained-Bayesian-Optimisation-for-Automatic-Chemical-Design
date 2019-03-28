"""
This module provides a baseline implementation of random sampling on the Branin Hoo function to compare against
Bayesian Optimisation.
"""

import argparse
import sys

import numpy as np
from numpy import genfromtxt
import scipy.stats as sps

from Branin_Sampler import branin  # Definition of Branin-Hoo function
from black_box_alpha import BB_alpha
from Diagnostic_Plots import best_so_far, GP_contours, BNN_contours, initial_data
from sparse_gp import SparseGP
from Constrained_BO.utils import load_object, save_object, load_data


def main(input_directory, output_directory):

    """

    :param input_directory: directory to which the output of Branin_Sampler.py was saved.
    :param output_directory: directory in which to save the plots.
    """

    np.random.seed(2)

    # Load the dataset

    X_bran = genfromtxt(input_directory + '/inputs.csv', delimiter=',', dtype='float32')
    y_con = genfromtxt(input_directory + '/constraint_targets.csv', delimiter=',', dtype='int')
    y_reg = genfromtxt(input_directory + '/branin_targets.csv', delimiter=',', dtype='float32')
    y_reg = y_reg.reshape((-1, 1))

    # We convert constraint targets from one-hot to categorical.

    y_con_cat = np.zeros(len(y_con), dtype=int)
    i = 0

    for element in y_con:
        if element[0] == 1:
            y_con_cat[i] = 1
        else:
            y_con_cat[i] = 0
        i += 1

    y_con = y_con_cat

    n_bran = X_bran.shape[0]  # number of examples

    permutation = np.random.choice(n_bran, n_bran, replace=False) # We shuffle the data

    X_tr_bran = X_bran[permutation, :][40: np.int(np.round(0.9 * n_bran)), :]  # 50/10 train/test split.
    X_te_bran = X_bran[permutation, :][np.int(np.round(0.8 * n_bran)): np.int(np.round(0.9 * n_bran)), :]

    y_tr_reg = y_reg[permutation][40: np.int(np.round(0.9 * n_bran))]  # 10:20 have balanced class split after the permutation is applied with random seed = 1
    y_te_reg = y_reg[permutation][np.int(np.round(0.8 * n_bran)): np.int(np.round(0.9 * n_bran))]
    y_tr_con = y_con[permutation][40: np.int(np.round(0.9 * n_bran))]  # no test set for constraint as traning subroutine for BNN doesn't require it
    y_te_con = y_con[permutation][np.int(np.round(0.8 * n_bran)): np.int(np.round(0.9 * n_bran))]

    # We plot the data used to initialise the surrogate model

    X1 = X_tr_bran[:, 0]
    X2 = X_tr_bran[:, 1]

    save_object(X1, output_directory + "/X1.dat")
    save_object(X2, output_directory + "/X2.dat")

    # We store the best feasible value found in the training set for reference

    feasible_vals = []

    for i in range(X_tr_bran.shape[0]):

        if y_tr_con[i] == 0:
            continue

        feasible_vals.append([branin(tuple(X_tr_bran[i]))])

    best_tr = min(feasible_vals)
    best_tr = best_tr[0]

    save_object(best_tr, output_directory + "/best_feasible_training_point.dat")

    # We set the number of data colletion iterations

    num_iters = 4

    for iteration in range(num_iters):

        # We train the regression model

        # We fit the GP

        # M = np.int(np.maximum(10,np.round(0.1 * n_bran)))

        M = 20

        sgp = SparseGP(X_tr_bran, 0 * X_tr_bran, y_tr_reg, M)
        sgp.train_via_ADAM(X_tr_bran, 0 * X_tr_bran, y_tr_reg, X_te_bran, X_te_bran * 0,
                           y_te_reg, minibatch_size=M, max_iterations=400, learning_rate=0.005)

        save_object(sgp, output_directory + "/sgp{}.dat".format(iteration))

        # We load the saved gp

        sgp = load_object(output_directory + "/sgp{}.dat".format(iteration))

        # We load some previous trained gp

        pred, uncert = sgp.predict(X_te_bran, 0 * X_te_bran)
        error = np.sqrt(np.mean((pred - y_te_reg)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_te_reg, scale=np.sqrt(uncert)))
        print('Test RMSE: ', error)
        print('Test ll: ', testll)

        pred, uncert = sgp.predict(X_tr_bran, 0 * X_tr_bran)
        error = np.sqrt(np.mean((pred - y_tr_reg)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_tr_reg, scale=np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)

        # we train the constraint network

        # We load the random seed

        seed = 1
        np.random.seed(seed)

        # We load the data

        datasets, n, d, n_labels = load_data(X_tr_bran, y_tr_con, X_te_bran, y_te_con)

        train_set_x, train_set_y = datasets[0]
        test_set_x, test_set_y = datasets[1]

        N_train = train_set_x.get_value(borrow=True).shape[0]
        N_test = test_set_x.get_value(borrow=True).shape[0]
        layer_sizes = [d, 50, n_labels]
        n_samples = 50
        alpha = 0.5
        learning_rate = 0.001
        v_prior = 1.0
        batch_size = 10
        print('... building model')
        sys.stdout.flush()
        bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size,
                            train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test)
        print('... training')
        sys.stdout.flush()

        test_error, test_ll = bb_alpha.train(400)

        # We save the trained BNN

        sys.setrecursionlimit(4000)  # Required to save the BNN

        save_object(bb_alpha, output_directory + "/bb_alpha{}.dat".format(iteration))

        # We pick the next 5 inputs based on random sampling

        np.random.seed()

        num_inputs = 1

        x1 = np.random.uniform(-5, 10, size=num_inputs)
        x2 = np.random.uniform(0, 15, size=num_inputs)
        random_inputs = np.zeros([num_inputs, 2])
        random_inputs[:, 0] = x1
        random_inputs[:, 1] = x2

        reg_scores = []  # collect y-values for Branin-Hoo function
        con_scores = []  # collect y-values for Constraint function
        probs = []  # collect the probabilities of satisfying the constraint
        log_probs = []  # collect the log probabilities of satisfying the constraint

        for i in range(random_inputs.shape[0]):

            reg_scores.append([branin(tuple(random_inputs[i]))])

            if (random_inputs[i][0] - 2.5)**2 + (random_inputs[i][1] - 7.5)**2 <= 50:
                con_scores.append(np.int64(1))
            else:
                con_scores.append(np.int64(0))

            probs.append(bb_alpha.prediction_probs(random_inputs[i].reshape(1, d))[0][0][1])
            log_probs.append(bb_alpha.pred_log_probs(random_inputs[i].reshape(1, d))[0][0][1])

            print(i)

        # print the value of the Branin-Hoo function at the data points we have acquired

        print(reg_scores)

        # save y-values and (x1,x2)-coordinates of locations chosen for evaluation

        save_object(reg_scores, output_directory + "/scores{}.dat".format(iteration))
        save_object(random_inputs, output_directory + "/next_inputs{}.dat".format(iteration))
        save_object(con_scores, output_directory + "/con_scores{}.dat".format(iteration))
        save_object(probs, output_directory + "/probs{}.dat".format(iteration))
        save_object(log_probs, output_directory + "/log_probs{}.dat".format(iteration))

        # extend labelled training data for next cycle

        X_tr_bran = np.concatenate([X_tr_bran, random_inputs], 0)
        y_tr_reg = np.concatenate([y_tr_reg, np.array(reg_scores)], 0)
        y_tr_con = np.concatenate([y_tr_con, np.array(con_scores)], 0)

    best_so_far(output_directory, num_iters)  # Plot the best point as a function of the data collection iteration number
    GP_contours(output_directory, num_iters)  # Plot the contours of the GP regression model
    BNN_contours(output_directory, num_iters)  # Plot the contours of the BNN constraint model
    initial_data(output_directory)  # Plot the data used to initialise the model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Sampling of the Branin Hoo function.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_directory", nargs='?', help="the directory containing samples from the Branin-Hoo function", default='joint_sample_data')
    parser.add_argument("output_directory", nargs='?', help='the output directory to which plots are saved', default='14_Dec_2018_Random')
    args = parser.parse_args()

    if args.output_directory is None:
        print('Using default output directory because none was supplied')
    if args.input_directory is None:
        print('Using the default input directory because none was supplied')
    main(args.input_directory, args.output_directory)
