"""
Script for partitioning the classification training data into positive and
negative class labels.
"""

import numpy as np

from Constrained_BO.utils import load_object, save_object

# We load the randomly sampled data from the grid defined by the lowest and highest values
# in the training data

x_samp = np.loadtxt('Collated_Data/X_sampled.txt')

# We load the training data

x_latent_feat = np.loadtxt('Collated_Data/latent_faetures.txt')

# We collect the negative class latent points.
# i is the loop iterator for the different criterion of which there are 10
# An example of one criterion is:

#    - y_con_5 means 5% of decodings must be valid to get a positive label

#    - The list of 10 criteria correspond to [5%, 10%, 20%, 30%, 40%, ... , 90%]

# The following loop will construct 10 training sets corresponding to the above
# criteria

i = 1

criterion = 5

while i <= 10:

    crit_string = 'y_con_{}.dat'.format(criterion)

    # list for storing latent points

    set_store = []

    # set start and end_count according to numbers in the Collated data folder

    start_count = 1
    end_count = 6

    for n in range(start_count, end_count):

        # open the class labels for the sampled points

        labels = load_object('Collated_Data/N{}/'.format(n) + '{}'.format(crit_string))
        length = len(labels)

        # Extract the indices that correspond to negative class

        neg_indexes = [k for k in range(length) if labels[k] == 0]

        # store the number of data points negative class labels in the sample

        num_negs = len(neg_indexes)

        # retrieve a list of latent points for which the class assigned was negative

        list_latent_points = [x_samp[m] for m in neg_indexes]

        # extract the points from the list and convert to a numpy array

        j = 0

        for point in list_latent_points:
            if j == 0:
                concat = point
                j += 1
            else:
                concat = np.concatenate((concat, point))

        # reshape to correct dimensions

        latent_points = concat.reshape(num_negs, 56)

        set_store.append(latent_points)

    h = 0

    for points in set_store:
        if h == 0:
            X_con_tr = points
            h += 1
        else:
            X_con_tr = np.concatenate((X_con_tr, points))

    # Save the latent points and corresponding labels (0 for negative case)

    num_examples = X_con_tr.shape[0]
    y_con_tr = np.zeros([num_examples])

    save_object(X_con_tr, 'train_test_sets/N_40000_Samples/40000_Neg_X_con_tr_{}.dat'.format(criterion))
    save_object(y_con_tr, 'train_test_sets/N_40000_Samples/40000_Neg_y_con_tr_{}.dat'.format(criterion))
    save_object(num_examples, 'train_test_sets/N_40000_Samples/40000_num_examples_{}.dat'.format(criterion))

    if i == 1:
        criterion += 5
    else:
        criterion += 10

    i += 1

# We collect the positive class latent points

o = 1

criterion = 5

while o <= 10:

    crit_string = 'y_con_{}.dat'.format(criterion)

    # list for storing latent points

    set_store = []

    # set start and end_count according to numbers in the Collated data folder

    start_count = 1
    end_count = 10

    for n in range(start_count, end_count):

        # open the class labels for the sampled points

        labels = load_object('Collated_Data/P{}/'.format(n) + '{}'.format(crit_string))
        length = len(labels)

        # Extract the indices that correspond to positive class

        pos_indexes = [k for k in range(length) if labels[k] == 1]

        # store the number of data points positive class labels in the sample

        num_pos = len(pos_indexes)

        # retrieve a list of latent points for which the class assigned was positive

        list_latent_points = [x_latent_feat[m] for m in pos_indexes]

        # extract the points from the list and convert to a numpy array

        j = 0

        for point in list_latent_points:
            if j == 0:
                concat = point
                j += 1
            else:
                concat = np.concatenate((concat, point))

        # reshape to correct dimensions

        latent_points = concat.reshape(num_pos, 56)

        set_store.append(latent_points)

    h = 0

    for points in set_store:
        if h == 0:
            X_con_tr = points
            h += 1
        else:
            X_con_tr = np.concatenate((X_con_tr, points))

    # Save the latent points and corresponding labels (1 for positive case)

    num_examples = X_con_tr.shape[0]
    y_con_tr = np.ones([num_examples])

    save_object(X_con_tr, 'train_test_sets/P_80000_Samples/80000_Pos_X_con_tr_{}.dat'.format(criterion))
    save_object(y_con_tr, 'train_test_sets/P_80000_Samples/80000_Pos_y_con_tr_{}.dat'.format(criterion))
    save_object(num_examples, 'train_test_sets/P_80000_Samples/80000_num_examples_{}.dat'.format(criterion))

    if o == 1:
        criterion += 5
    else:
        criterion += 10

    o += 1
