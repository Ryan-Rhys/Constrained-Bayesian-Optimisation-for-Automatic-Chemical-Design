"""
Module containing functions for plotting.
"""

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pylab
from pylab import MaxNLocator

from Constrained_BO.utils import load_object, save_object


def my_polygon_scatter(axes, x_array, y_array, resolution=5, radius=0.5, **kwargs):
    ''' resolution is number of sides of polygon '''
    for x, y in zip(x_array, y_array):
        polygon = matplotlib.patches.CirclePolygon((x, y), radius=radius, resolution=resolution, **kwargs)
        axes.add_patch(polygon)
    return True


def initial_data(results_directory):
    """
    Produces a scatterplot of the data used to initialise the surrogate model.

    :param results_directory: the directory containing the (x1, x2) data.
    """

    X1 = load_object(results_directory + "/X1.dat")
    X2 = load_object(results_directory + "/X2.dat")
    plt.figure(1)
    plt.title('Initial Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(X1, X2)
    pylab.savefig(results_directory + "/initial_data.png")


def best_so_far(results_directory, num_iterations):
    """
    Function that plots:

        1) The best feasible value obtained so far as a function of the number of iterations
        2) A scatterplot showing the data points collected

    :param results_directory: directory to save the plots to.
    :param num_iterations: the number of iterations for which data collection is being carried out.
    """

    best_vals = []

    # coordinates of collected data points

    x1_vals = []
    x2_vals = []
    counter = 0
    first_find = 0

    for iteration in range(num_iterations):

        # We monitor the best value obtained so far

        evaluations = load_object(results_directory + "/scores{}.dat".format(iteration))
        best_value = min(evaluations)
        constraint_value = load_object(results_directory + "/con_scores{}.dat".format(iteration))

        # We DON'T use the best value found in the training data if the first collected point is not feasible

        if constraint_value[0] == 1 and counter == 0:
            counter += 1
            best_vals.append(best_value[0])
            first_find += 1

        if counter > 0:
            if first_find == 1:
                first_find += 1
            else:
                counter += 1
                if best_value[0] < min(best_vals):
                    best_vals.append(best_value[0])
                else:
                    best_vals.append(min(best_vals))

        # We collect the data points for plotting

        next_inputs = load_object(results_directory + "/next_inputs{}.dat".format(iteration))

        for data_point in next_inputs:
            x1_vals.append(data_point[0])
            x2_vals.append(data_point[1])

    iterations = range((num_iterations - counter) + 1, num_iterations + 1)

    # We plot the best value obtained so far as a function of iterations

    plt.figure(2)
    axes = plt.figure(2).gca()
    xa, ya = axes.get_xaxis(), axes.get_yaxis()
    xa.set_major_locator(MaxNLocator(integer=True)) # force axis ticks to be integers
    ya.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((num_iterations - counter) + 1, num_iterations)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Feasible Value')
    plt.plot(iterations, best_vals)
    pylab.savefig(results_directory + "/best_so_far.png")
    plt.close()

    save_object(iterations, results_directory + "/iterations.dat")
    save_object(best_vals, results_directory + "/best_vals.dat")

    # We plot the data points collected

    plt.figure(3)
    plt.title('Data Points Collected')
    plt.gca().set_aspect('equal')
    plt.xlim(-5, 10)
    plt.ylim(0, 15)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x1_vals, x2_vals)
    pylab.savefig(results_directory + "/data_collected.png")
    plt.close()


def GP_contours(results_directory, num_iterations):

    """
    Function that plots:

        1) The predictive mean of the GP regression model
        2) The variance of the GP regression model

    :param results_directory: the directory in which the plots are saved.
    :param num_iterations: the number of iterations for which data collection is carried out.
    """

    # We load the saved GP regression model

    sgp = load_object(results_directory + "/sgp{}.dat".format(num_iterations - 1))

    # We prepare the contour grid

    delta = 0.025  # grid spacing
    x = np.arange(-5.0, 10.0, delta)
    y = np.arange(0.0, 15.0, delta)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(len(x)**2, 1)
    Y = Y.reshape(len(y)**2, 1)

    # We reshape the meshgrid in a way such that it can be passed into the sgp and bnn prediction functions

    reshaped_grid = np.zeros([len(x)**2, 2])
    reshaped_grid[:, 0] = X.reshape(len(x)**2)
    reshaped_grid[:, 1] = Y.reshape(len(x)**2, order='F')

    # We plot the predictive mean and variance of the GP regression model

    pred, uncert = sgp.predict(reshaped_grid, 0 * reshaped_grid)
    branin, uncert = pred.reshape(len(x), len(x)), uncert.reshape(len(x), len(x))
    plt.figure(4)
    plt.gca().set_aspect('equal', adjustable='box')
    CS = plt.contourf(x, y, branin, cmap=cm.viridis_r)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    axes = plt.gca()
    my_polygon_scatter(axes, [np.pi], [2.275], radius=.5, resolution=3, alpha=.5, color='r')
    pylab.savefig(results_directory + "/branin_contour.png")
    plt.close()

    plt.figure(5)
    plt.gca().set_aspect('equal', adjustable='box')
    Cs = plt.contourf(x, y, uncert, cmap=cm.viridis_r)
    Cb = plt.colorbar(Cs, shrink=0.8, extend='both')
    axes = pylab.axes()
    my_polygon_scatter(axes, [np.pi], [2.275], radius=.5, resolution=3, alpha=.5, color='r')
    pylab.savefig(results_directory + "/branin_uncertainty.png")
    plt.close()


def BNN_contours(results_directory, num_iterations):

    """
    Function that plots:

        1) The positive class probabilities of the BNN logistic regression model

    :param results_directory: the directory in which plots are saved.
    :param num_iterations: the number of iterations for which data collection is carried out.
    """

    # We load the saved BNN logistic regression model

    bnn = load_object(results_directory + "/bb_alpha{}.dat".format(num_iterations - 1))

    # We prepare the contour grid

    delta = 0.05 # grid spacing
    x = np.arange(-5.0, 10.0, delta)
    y = np.arange(0.0, 15.0, delta)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(len(x)**2, 1)
    Y = Y.reshape(len(y)**2, 1)

    # We reshape the meshgrid in a way such that it can be passed into the sgp and bnn prediction functions

    reshaped_grid = np.zeros([len(x)**2, 2])
    reshaped_grid[:, 0] = X.reshape(len(x)**2)
    reshaped_grid[:, 1] = Y.reshape(len(x)**2, order='F')

    # We plot the constraint probabilities of the BNN logistic regression model
    # which should resemble the disk constraint

    import sys; sys.stdout.flush()
    probs_array = bnn.prediction_probs(reshaped_grid)
    positive_class_probs = probs_array[0][:, 1]
    constraint = positive_class_probs.reshape(len(x), len(x))

    plt.figure(6)
    plt.gca().set_aspect('equal', adjustable='box')
    Css = plt.contourf(x, y, constraint, np.arange(0, 1, .1), extend='both')
    Cbb = plt.colorbar(Css, shrink=0.8, extend='both')
    axes=pylab.axes()
    my_polygon_scatter(axes, [np.pi], [2.275], radius=.5, resolution=3, alpha=.5, color='r')

    pylab.savefig(results_directory + "/constraint_contour.png")
    plt.close()
