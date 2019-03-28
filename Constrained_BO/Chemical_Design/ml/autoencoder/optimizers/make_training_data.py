"""
This module constructs a training set consisting of molecules collected
by decoding the data on which the autoencoder was trained.
"""

import numpy as np

from Constrained_BO.utils import load_object, save_object

x_latent_feat = np.loadtxt('Collated_Data/latent_faetures.txt')


def make_training_data(X_train, num_valid_decodings, start_count, end_count):
    """
    Function that makes a training data set for binary classification
    of validity based on the number of valid decodings from 100 attempts.
    Assumes that data folders to look through are of the format P1, P2, P3,...

    :param X_train: Latent features of the training data for the autoencoder.
    :param num_valid_decodings: an int between 0 and 100 representing the
    threshold for classification as valid or invalid.
    :param start_count: the index of the first data folder to look through!
    :param end_count: the index of the last data folder to look through
    :return A training set of (x,y) pairs for binary classification.
    """

    validity_criterion_string = 'y_con_{}.dat'.format(num_valid_decodings)

    for i in range(start_count, end_count):

        labels = load_object('Collated_Data/P{}/'.format(i) +
                             '{}'.format(validity_criterion_string))
        num_labels = len(labels)

        pos_indices = [p for p in range(num_labels) if labels[p] == 1]
        neg_indices = [n for n in range(num_labels) if labels[n] == 0]

        num_pos_labels = len(pos_indices)
        num_neg_labels = len(neg_indices)

        assert num_pos_labels + num_neg_labels == num_labels

        pos_latent_features_list = [X_train[p] for p in pos_indices]
        neg_latent_features_list = [X_train[n] for n in neg_indices]

        X_con_tr_pos = np.array(pos_latent_features_list)
        X_con_tr_neg = np.array(neg_latent_features_list)

        assert X_con_tr_pos.shape == (num_pos_labels, 56)
        assert X_con_tr_neg.shape == (num_neg_labels, 56)

        y_con_tr_pos = np.ones([num_pos_labels])
        y_con_tr_neg = np.zeros([num_neg_labels])

        assert y_con_tr_pos.shape == (num_pos_labels,)
        assert y_con_tr_neg.shape == (num_neg_labels,)

        if i == start_count:
            X_con_tr_full_pos = X_con_tr_pos
            X_con_tr_full_neg = X_con_tr_neg
            y_con_tr_full_pos = y_con_tr_pos
            y_con_tr_full_neg = y_con_tr_neg
        else:
            X_con_tr_full_pos = np.concatenate((X_con_tr_full_pos, X_con_tr_pos))
            X_con_tr_full_neg = np.concatenate((X_con_tr_full_neg, X_con_tr_neg))
            y_con_tr_full_pos = np.concatenate((y_con_tr_full_pos, y_con_tr_pos))
            y_con_tr_full_neg = np.concatenate((y_con_tr_full_neg, y_con_tr_neg))

    num_pos_examples = X_con_tr_full_pos.shape[0]
    num_neg_examples = X_con_tr_full_neg.shape[0]

    save_object(X_con_tr_full_pos,
                'train_test_sets/Train_Samples/Positive_Latents/X_con_tr_pos{}.dat'.format(num_valid_decodings))
    save_object(y_con_tr_full_pos,
                'train_test_sets/Train_Samples/Positive_Latents/Y_con_tr_pos{}.dat'.format(num_valid_decodings))
    save_object(num_pos_examples,
                'train_test_sets/Train_Samples/Positive_Latents/num_pos_examples{}.dat'.format(num_valid_decodings))
    save_object(X_con_tr_full_neg,
                'train_test_sets/Train_Samples/Negative_Latents/X_con_tr_neg{}.dat'.format(num_valid_decodings))
    save_object(y_con_tr_full_neg,
                'train_test_sets/Train_Samples/Negative_Latents/Y_con_tr_neg{}.dat'.format(num_valid_decodings))
    save_object(num_neg_examples,
                'train_test_sets/Train_Samples/Negative_Latents/num_pos_examples{}.dat'.format(num_valid_decodings))

    return None


if __name__ == "__main__":
    X_train = np.loadtxt('Collated_Data/latent_faetures.txt')
    num_valid_decodings = 20
    start_count = 1
    end_count = 20
    make_training_data(X_train, num_valid_decodings, start_count, end_count)
