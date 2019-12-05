"""
Sample points uniformly at random from the latent space.
"""

import numpy as np

if __name__ == '__main__':

    num_samples = 300000  # Would suggest a big number. 300000 samples were taken for the experiments
    num_latent_dims = 56  # Dictated by the latent dimension of the VAE

    # We load the upper and lower bounds (limits) of each of the 56 dimensions in the latent space.

    max_lims = np.loadtxt('max_lims.txt')
    min_lims = np.loadtxt('min_lims.txt')

    # We sample uniformly at random from the hypercube dictated by these bounds

    sampled_points = np.random.uniform(min_lims, max_lims, (num_samples, num_latent_dims))
    np.savetxt('X_sampled.txt', sampled_points)
