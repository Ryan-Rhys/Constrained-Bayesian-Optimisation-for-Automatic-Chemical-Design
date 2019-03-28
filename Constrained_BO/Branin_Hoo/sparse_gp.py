"""
This class represents a node within the network.
"""

from __future__ import print_function

import sys
import time

import scipy.optimize as spo
import scipy.stats as sps

from Constrained_BO.sparse_gp_theano_internal import *
from Constrained_BO.utils import casting, LogSumExp


def global_optimization(grid, lower, upper, function_grid, function_scalar, function_scalar_gradient):

    grid_values = function_grid(grid)
    best = grid_values.argmin()

    # We solve the optimization problem

    X_initial = grid[best: (best + 1), :]
    def objective(X):
        X = casting(X)
        X = X.reshape((1, grid.shape[1]))
        value = function_scalar(X)
        gradient_value = function_scalar_gradient(X).flatten()

        return np.float(value), gradient_value.astype(np.float) 

    lbfgs_bounds = zip(lower.tolist(), upper.tolist())
    x_optimal, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, X_initial, bounds=lbfgs_bounds, iprint=0, maxiter=150)
    x_optimal = x_optimal.reshape((1, grid.shape[1]))

    return x_optimal, y_opt


def adam_theano(loss, all_params, learning_rate=0.001):
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(casting(1.0))
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * g                           # (Update biased first moment estimate)
        v = b2 * v_previous + (1 - b2) * g**2                            # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e)  #(Update parameters)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1.))
    return updates


class SparseGP:

    # The training_targets are the Y's which in the case of regression are real numbers in the case of binary
    # classification are 1 or -1 and in the case of multiclass classification are 0, 1, 2,.. n_class - 1

    def __init__(self, input_means, input_vars, training_targets, n_inducing_points):

        self.input_means_numpy = input_means
        self.input_means = theano.shared(value=input_means.astype(theano.config.floatX), borrow=True, name='X')
        self.input_vars = theano.shared(value=input_vars.astype(theano.config.floatX), borrow=True, name='X')
        self.original_training_targets = theano.shared(value=training_targets.astype(theano.config.floatX), borrow=True, name='y')
        self.training_targets = self.original_training_targets

        self.n_points = input_means.shape[0]
        self.d_input = input_means.shape[1]

        self.sparse_gp = Sparse_GP(n_inducing_points, self.n_points, self.d_input, self.input_means, self.input_vars, self.training_targets)

        self.set_for_prediction = False
        self.predict_function = None

    def initialize(self):
        self.sparse_gp.initialize()

    def setForTraining(self):
        self.sparse_gp.setForTraining()

    def setForPrediction(self):
        self.sparse_gp.setForPrediction()

    def get_params(self):
        return self.sparse_gp.get_params()

    def set_params(self, params):
        self.sparse_gp.set_params(params)

    def getEnergy(self):
        self.sparse_gp.compute_output()
        return self.sparse_gp.getContributionToEnergy()[0, 0]

    def predict(self, means_test, vars_test):

        self.setForPrediction()

        means_test = means_test.astype(theano.config.floatX)
        vars_test = vars_test.astype(theano.config.floatX)

        if self.predict_function is None:

            self.sparse_gp.compute_output()
            predictions = self.sparse_gp.getPredictedValues()

            X = T.matrix('X', dtype=theano.config.floatX)
            Z = T.matrix('Z', dtype=theano.config.floatX)

            self.predict_function = theano.function([X, Z], predictions, givens={self.input_means: X, self.input_vars: Z})

        predicted_values = self.predict_function(means_test, vars_test)

        self.setForTraining()

        return predicted_values

    # This trains the network via LBFGS as implemented in scipy (slow but good for small datasets)

    def train_via_LBFGS(self, input_means, input_vars, training_targets, max_iterations=500):

        # We initialize the network and get the initial parameters

        input_means = input_means.astype(theano.config.floatX)
        input_vars = input_vars.astype(theano.config.floatX)
        training_targets = training_targets.astype(theano.config.floatX)
        self.input_means.set_value(input_means)
        self.input_vars.set_value(input_vars)
        self.original_training_targets.set_value(training_targets)

        self.initialize()
        self.setForTraining()

        X = T.matrix('X', dtype=theano.config.floatX)
        Z = T.matrix('Z', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)
        e = self.getEnergy()
        energy = theano.function([X, Z, y], e, givens={self.input_means: X, self.input_vars: Z, self.training_targets: y})
        all_params = self.get_params()
        energy_grad = theano.function([X, Z, y], T.grad(e, all_params),
                                      givens={self.input_means: X, self.input_vars: Z, self.training_targets: y})

        initial_params = theano.function([], all_params)()

        params_shapes = [s.shape for s in initial_params]

        def de_vectorize_params(params):
            ret = []
            for shape in params_shapes:
                if len(shape) == 2:
                    ret.append(params[: np.prod(shape)].reshape(shape))
                    params = params[np.prod(shape):]
                elif len(shape) == 1:
                    ret.append(params[: np.prod(shape)])
                    params = params[np.prod(shape):]
                else:
                    ret.append(params[0])
                    params = params[1:]
            return ret

        def vectorize_params(params):
            return np.concatenate([s.flatten() for s in params])

        def objective(params):

            params = de_vectorize_params(params)
            self.set_params(params)
            energy_value = energy(input_means, input_vars, training_targets)
            gradient_value = energy_grad(input_means, input_vars, training_targets)

            return -energy_value, -vectorize_params(gradient_value)

        # We create a theano function that evaluates the energy

        initial_params = vectorize_params(initial_params)
        x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, initial_params, bounds=None, iprint=1, maxiter=max_iterations)

        self.set_params(de_vectorize_params(x_opt))

        return y_opt

    def train_via_ADAM(self, input_means, input_vars, training_targets, input_means_test, input_vars_test, test_targets,
                       max_iterations=500, minibatch_size=4000, learning_rate=1e-3):

        input_means = input_means.astype(theano.config.floatX)
        input_vars = input_vars.astype(theano.config.floatX)
        training_targets = training_targets.astype(theano.config.floatX)
        n_data_points = input_means.shape[0]
        selected_points = np.random.choice(n_data_points, n_data_points, replace=False)[0: min(n_data_points, minibatch_size)]
        self.input_means.set_value(input_means[selected_points, :])
        self.input_vars.set_value(input_vars[selected_points, :])
        self.original_training_targets.set_value(training_targets[selected_points, :])

        print('Initializing network')
        sys.stdout.flush()
        self.setForTraining()
        self.initialize()

        X = T.matrix('X', dtype=theano.config.floatX)
        Z = T.matrix('Z', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)

        e = self.getEnergy()

        all_params = self.get_params()

        print('Compiling adam updates')
        sys.stdout.flush()

        process_minibatch_adam = theano.function([X, Z, y], -e, updates=adam_theano(-e, all_params, learning_rate),
            givens={self.input_means: X, self.input_vars: Z, self.original_training_targets: y})

        # Main loop of the optimization

        print('Training')
        sys.stdout.flush()
        n_batches = int(np.ceil(1.0 * n_data_points / minibatch_size))
        for j in range(max_iterations):
            suffle = np.random.choice(n_data_points, n_data_points, replace=False)
            input_means = input_means[suffle, :]
            input_vars = input_vars[suffle, :]
            training_targets = training_targets[suffle, :]

            for i in range(n_batches):
                minibatch_data_means = input_means[i * minibatch_size: min((i + 1) * minibatch_size, n_data_points), :]
                minibatch_data_vars = input_vars[i * minibatch_size: min((i + 1) * minibatch_size, n_data_points), :]
                minibatch_targets = training_targets[i * minibatch_size: min((i + 1) * minibatch_size, n_data_points), :]

                start = time.time()
                current_energy = process_minibatch_adam(minibatch_data_means, minibatch_data_vars, minibatch_targets)
                elapsed_time = time.time() - start

                print('Epoch: {}, Mini-batch: {} of {} - Energy: {} Time: {}'.format(j, i, n_batches, current_energy, elapsed_time))
                sys.stdout.flush()

            pred, uncert = self.predict(input_means_test, input_vars_test)
            test_error = np.sqrt(np.mean((pred - test_targets)**2))
            test_ll = np.mean(sps.norm.logpdf(pred - test_targets, scale=np.sqrt(uncert)))

            print('Test error: {} Test ll: {}'.format(test_error, test_ll))
            sys.stdout.flush()

            pred = np.zeros((0, 1))
            uncert = np.zeros((0, uncert.shape[1]))
            for i in range(n_batches):
                minibatch_data_means = input_means[i * minibatch_size: min((i + 1) * minibatch_size, n_data_points), :]
                minibatch_data_vars = input_vars[i * minibatch_size: min((i + 1) * minibatch_size, n_data_points), :]
                pred_new, uncert_new = self.predict(minibatch_data_means, minibatch_data_vars)
                pred = np.concatenate((pred, pred_new), 0)
                uncert = np.concatenate((uncert, uncert_new), 0)

            training_error = np.sqrt(np.mean((pred - training_targets)**2))
            training_ll = np.mean(sps.norm.logpdf(pred - training_targets, scale=np.sqrt(uncert)))

            print('Train error: {} Train ll: {}'.format(training_error, training_ll))
            sys.stdout.flush()

    def get_incumbent(self, bb_alpha, grid, bb_alpha_samples):

        self.sparse_gp.compute_output()
        m, v = self.sparse_gp.getPredictedValues() 

        X = T.matrix('X', dtype=theano.config.floatX)
        pred_probs = T.exp(LogSumExp(bb_alpha.network.output(X), 0) + T.log(1.0/bb_alpha_samples))

        function_grid = theano.function([X], m, givens={self.input_means: X, self.input_vars: 0 * X})
        function_grid_prob = theano.function([X], T.reshape(pred_probs[:, :, 1], [T.shape(X)[0], 1]), givens={self.input_means: X, self.input_vars: 0 * X})

        m_on_grid = function_grid(grid)
        p_on_grid = function_grid_prob(grid)

        # obtain row in grid for which m_on_grid is smallest subject to p_on_grid larger than 0.95
        # if all p_on_grid smaller than 0.95 then obtain row in grid for which p_on_grid is the largest

        if np.max(p_on_grid) < 0.95:
            grid_row_val = grid[np.argmax(p_on_grid)]

        else:
            feasible_point_indices = [i for i in range(len(p_on_grid)) if p_on_grid[i] >= 0.95]
            grid_row_val = grid[[i for i in feasible_point_indices if m_on_grid[i] == max([m_on_grid[i] for i in feasible_point_indices])][0]]

        # return value of function function_grid evaluated on the resulting row from grid
        # me: returns array of shape [1,1] for the incumbent best point.

        return function_grid(grid_row_val.reshape(1, 2))

    def optimize_ei(self, bb_alpha, grid, lower, upper, incumbent, bb_alpha_samples):

        X = T.matrix('X', dtype=theano.config.floatX)
        log_ei = self.sparse_gp.compute_log_ei(X, incumbent)
        pred_log_probs = LogSumExp(bb_alpha.network.output(X), 0) + T.log(1.0 / bb_alpha_samples)

        function_grid = theano.function([X], -log_ei - T.reshape(pred_log_probs[:, :, 1], [T.shape(X)[0], 1]))
        function_scalar = theano.function([X], -log_ei[0, 0] - T.reshape(pred_log_probs[:, :, 1], [T.shape(X)[0], 1])[0, 0])
        function_scalar_gradient = theano.function([X], -T.grad(log_ei[0, 0] + T.reshape(pred_log_probs[:, :, 1], [T.shape(X)[0], 1])[0, 0], X))

        return global_optimization(grid, lower, upper, function_grid, function_scalar, function_scalar_gradient)[0]

    def batched_greedy_ei(self, bb_alpha, q, lower, upper, bb_alpha_samples, n_samples=1):
        """
        Subroutine to select data points subject to constraint

        bb_alpha: instance of BB_Alpha class

        """

        # 28 July 20:09 - Note that the log_ei defined here has different dimenionsionality to that defined in sparse_gp_theano_internal.py
        # it's being returned as log_ei[:, 0] in sparse_gp_theano_internal

        self.setForPrediction()

        grid_size = 10000
        grid = casting(lower + np.random.rand(grid_size, self.d_input) * (upper - lower))
        grid = np.concatenate([grid, self.input_means_numpy], 0)  # 28 July new line added to put the training data with the grid.

        incumbent = self.get_incumbent(bb_alpha, grid, bb_alpha_samples)
        X_numpy = self.optimize_ei(bb_alpha, grid, lower, upper, incumbent, bb_alpha_samples)
        randomness_numpy = casting(0 * np.random.randn(X_numpy.shape[0], n_samples).astype(theano.config.floatX))

        randomness = theano.shared(value=randomness_numpy.astype(theano.config.floatX), name='randomness', borrow=True)
        X = theano.shared(value=X_numpy.astype(theano.config.floatX), name='X', borrow=True)
        x = T.matrix('x', dtype=theano.config.floatX)

        log_ei = self.sparse_gp.compute_log_averaged_ei(x, X, incumbent)
        pred_log_probs = LogSumExp(bb_alpha.network.output(x), 0) + T.log(1.0 / bb_alpha_samples)  # 2-D array of size (n_samples, 2) where column 1 gives the log probability of the constraint being unsatisfied and column two gives the log probabilty of the constraint being satisfied

        function_grid = theano.function([x], -log_ei - T.reshape(pred_log_probs[:, :, 1], [T.shape(x)[0], 1])[:, 0])  # indices for pred_log_probs give the log probability of the constraint being satisfied
        function_scalar = theano.function([x], -log_ei[0] - T.reshape(pred_log_probs[:, :, 1], [T.shape(x)[0], 1])[0, 0])
        function_scalar_gradient = theano.function([x], -T.grad(log_ei[0] + T.reshape(pred_log_probs[:, :, 1], [T.shape(x)[0], 1])[0, 0], x))

        # We optimize the ei in a greedy manner

        for i in range(1, q): 

            new_point = global_optimization(grid, lower, upper, function_grid, function_scalar, function_scalar_gradient)[0]
            X_numpy = casting(np.concatenate([X_numpy, new_point], 0))
            randomness_numpy = casting(0 * np.random.randn(X_numpy.shape[0], n_samples).astype(theano.config.floatX))
            X.set_value(X_numpy)
            randomness.set_value(randomness_numpy)
            print(i, X_numpy)

        m, v = self.predict(X_numpy, 0 * X_numpy)

        print("Predictive mean at selected points:\n", m)

        return X_numpy
