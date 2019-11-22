"""
Module to generate negatively labelled (invalid) molecules from encoded latent points.
"""

from datetime import datetime
import numpy as np

from Constrained_BO.Chemical_Design.autoencoder.latent_space import encode_decode as lasp
from Constrained_BO.utils import save_object

# 19_July_Train_Data_Limits is randomly sampled data from the latent space where the bounds are given
# by the lowest and highest value for each dimension.

X_sample = np.loadtxt('19_July_Train_Data_Limits/X_sampled.txt')

# We decode a batch of sampled latent points from a grid limits defined by the highest
# and lowest values seen in the training data

X_batch = X_sample[83000:91000]

# We load the decoder to obtain the molecules

preproc = lasp.PreProcessing(dataset='drugs')
enc_dec = lasp.EncoderDecoder()
encoder, decoder = enc_dec.get_functions()

postprocessor = lasp.PostProcessing(enc_dec)

decode_attempts = 100

# We classify points according to the proportion of decode attempts that yield sensible molecules
# Sensible is defined as being a valid SMILES string with a minimum length of 5 characters

# We collect a list of the number of sensible decodings out of 100 for each latent point

num_sensible_per_point = []

# We collect lists of the decoded SMILES as a sanity check

valid_smiles_final = []
all_smiles_final = []

# We collect labels according to different decision rules
# y_con_5 means we classify a latent point as good if 5% of its 100 decode attempts
# result in sensible molecules

y_con_5 = np.zeros([X_batch.shape[0], ])
y_con_10 = np.zeros([X_batch.shape[0], ])
y_con_20 = np.zeros([X_batch.shape[0], ])
y_con_30 = np.zeros([X_batch.shape[0], ])
y_con_40 = np.zeros([X_batch.shape[0], ])
y_con_50 = np.zeros([X_batch.shape[0], ])
y_con_60 = np.zeros([X_batch.shape[0], ])
y_con_70 = np.zeros([X_batch.shape[0], ])
y_con_80 = np.zeros([X_batch.shape[0], ])
y_con_90 = np.zeros([X_batch.shape[0], ])

# We time the main bottleneck

startTime = datetime.now()

for i in range(X_batch.shape[0]):

    sampler_out = postprocessor.ls_to_smiles([X_batch[i: (i + 1), :]], decode_attempts, decode_attempts,)
    rdmols, valid_smiles, all_smiles, output_reps, distances = sampler_out

    valid_long_smiles = [x for x in valid_smiles if len(x) > 5]
    num_sensible = sum([all_smiles.count(x) for x in valid_long_smiles])  

    num_sensible_per_point.append(num_sensible)
    valid_smiles_final.append(valid_smiles)
    all_smiles_final.append(all_smiles)

    if num_sensible > 5:
        y_con_5[i] = 1.0
        if num_sensible > 10:
            y_con_10[i] = 1.0
            if num_sensible > 20:
                y_con_20[i] = 1.0
                if num_sensible > 30:
                    y_con_30[i] = 1.0
                    if num_sensible > 40:
                        y_con_40[i] = 1.0
                        if num_sensible > 50:
                            y_con_50[i] = 1.0
                            if num_sensible > 60:
                                y_con_60[i] = 1.0
                                if num_sensible > 70:
                                    y_con_70[i] = 1.0
                                    if num_sensible > 80:
                                        y_con_80[i] = 1.0
                                        if num_sensible > 90:
                                            y_con_90[i] = 1.0
                                        else:
                                            y_con_90[i] = 0.0
                                    else:
                                        y_con_80[i] = 0.0
                                else:
                                    y_con_70[i] = 0.0
                            else:
                                y_con_60[i] = 0.0
                        else:
                            y_con_50[i] = 0.0
                    else:
                        y_con_40[i] = 0.0        
                else:
                    y_con_30[i] = 0.0
            else:
                y_con_20[i] = 0.0
        else:
            y_con_10[i] = 0.0
    else:
        y_con_5[i] = 0.0 

    print(i)

# We save the labels
# See the directory Collated_Data/Data_Lexicon for a mapping between the training set indices and the directory number
# e.g. the directory number N10 corresponds to indices 83000-91000.
# The script make_training_data.py expects to see the directories in the P1 through P19 format.
# Create new directories N1, N2, ..., N14 in the Collated_Data folder as required

save_object(num_sensible_per_point, "Collated_Data/N10/num_sensible_per_point.dat")
save_object(valid_smiles_final, "Collated_Data/N10/valid_smiles.dat")
save_object(all_smiles_final, "Collated_Data/N10/all_smiles.dat")

save_object(y_con_5, "Collated_Data/N10/y_con_5.dat")
save_object(y_con_10, "Collated_Data/N10/y_con_10.dat")
save_object(y_con_20, "Collated_Data/N10/y_con_20.dat")
save_object(y_con_30, "Collated_Data/N10/y_con_30.dat")
save_object(y_con_40, "Collated_Data/N10/y_con_40.dat")
save_object(y_con_50, "Collated_Data/N10/y_con_50.dat")
save_object(y_con_60, "Collated_Data/N10/y_con_60.dat")
save_object(y_con_70, "Collated_Data/N10/y_con_70.dat")
save_object(y_con_80, "Collated_Data/N10/y_con_80.dat")
save_object(y_con_90, "Collated_Data/N10/y_con_90.dat")

print(datetime.now() - startTime)
