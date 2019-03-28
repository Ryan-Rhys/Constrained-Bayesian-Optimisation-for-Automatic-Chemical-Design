"""
Module for generating data points from the Branin-Hoo function.
"""

import argparse

import numpy as np


def branin(x):
    """

    :param x: 2-element list of [x1, x2] coordinates of the branin function branin(x1, x2).
    :return: branin-hoo function value.
    """

    x1 = x[0]
    x2 = x[1]
    a = 1.
    b = 5.1 / (4.*np.pi**2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8.*np.pi)
    ret = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s

    return ret

# Construct a balanced dataset for training


def main(output_directory, num_samples):

    """

    :param output_directory: the directory to which the data is saved.
    :param num_samples: the number of data points to collect from the function.
    """

    import math
    import random
    import csv

    # Positive Class Samples

    pos_x1_data = []
    pos_x2_data = []

    num_samples_range = range(0, num_samples)
    radius = math.sqrt(50)

    for i in num_samples_range:

        r, theta = [math.sqrt(random.uniform(0, radius))*math.sqrt(radius), 2*math.pi*random.random()]

        x1 = 2.5 + r * math.cos(theta)
        x2 = 7.5 + r * math.sin(theta)
        pos_x1_data.append(x1)
        pos_x2_data.append(x2)

    # Negative Class Samples

    neg_x1_data = []
    neg_x2_data = []

    j = 0

    while j < num_samples:

        x1, x2 = random.uniform(-5, 10), random.uniform(0, 15)

        if (x1-2.5)**2 + (x2-7.5)**2 > 50:
            neg_x1_data.append(x1)
            neg_x2_data.append(x2)
            j += 1

    # collect Branin-Hoo function values at the same inputs used to generate the constraint function values

    pos_examples = []
    neg_examples = []
    for i in range(0, len(pos_x1_data)):
        pos_examples.append((pos_x1_data[i], pos_x2_data[i], branin((pos_x1_data[i], pos_x2_data[i])), 1))
        neg_examples.append((neg_x1_data[i], neg_x2_data[i], branin((neg_x1_data[i], neg_x2_data[i])), 0))

    full_dataset = pos_examples + neg_examples
    random.shuffle(full_dataset)

    inputs = []
    constraint_targets = []
    branin_targets = []

    for j in range(0, len(full_dataset)):
        inputs.append([full_dataset[j][0], full_dataset[j][1]])
        if full_dataset[j][-1] == 1:
            constraint_targets.append([1, 0])
        else:
            constraint_targets.append([0, 1])
        branin_targets.append([full_dataset[j][2]])

    with open(output_directory + '/inputs.csv', 'wb') as myfile:
        for row in inputs:
            wr = csv.writer(myfile)
            wr.writerow(row)

    with open(output_directory + '/constraint_targets.csv', 'wb') as myfile2:
        for row in constraint_targets:
            wr = csv.writer(myfile2)
            wr.writerow(row)

    with open(output_directory + '/branin_targets.csv', 'wb') as myfile3:
        for row in branin_targets:
            wr = csv.writer(myfile3)
            wr.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimisation of the Branin Hoo function.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_directory", nargs='?', help='the output directory to which data are saved', default='joint_sample_data')
    parser.add_argument("num_samples", nargs='?', help='the number of samples to take from the branin function', default=100)
    args = parser.parse_args()

    if args.output_directory is None:
        print('Using default output directory because none was supplied')
    if args.num_samples is None:
        print('Using default num_samples of 100 because none was supplied')
    main(args.output_directory, args.num_samples)
