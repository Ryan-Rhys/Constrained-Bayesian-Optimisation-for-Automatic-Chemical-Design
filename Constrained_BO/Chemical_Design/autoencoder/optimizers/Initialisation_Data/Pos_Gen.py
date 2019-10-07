"""
Module to generate positively labelled (valid) molecules from encoded latent training data points.
"""

from datetime import datetime
import numpy as np

from Constrained_BO.Chemical_Design.autoencoder.latent_space import encode_decode as lasp
from Constrained_BO.utils import save_object


X = np.loadtxt('../latent_features_and_targets/latent_faetures.txt')

# We decode a batch of the training data
# The indices of 168000:173000 below are the latent features for the molecules indexed 168000:173000 in the Zinc
# dataset.

X_train = X[168000:173000]

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

y_con_5 = np.zeros([X_train.shape[0], ])
y_con_10 = np.zeros([X_train.shape[0], ])
y_con_20 = np.zeros([X_train.shape[0], ])
y_con_30 = np.zeros([X_train.shape[0], ])
y_con_40 = np.zeros([X_train.shape[0], ])
y_con_50 = np.zeros([X_train.shape[0], ])
y_con_60 = np.zeros([X_train.shape[0], ])
y_con_70 = np.zeros([X_train.shape[0], ])
y_con_80 = np.zeros([X_train.shape[0], ])
y_con_90 = np.zeros([X_train.shape[0], ])

# We time the main bottleneck

startTime = datetime.now()

for i in range(X_train.shape[0]):
    
    sampler_out = postprocessor.ls_to_smiles([X_train[i: (i + 1), :]], decode_attempts, decode_attempts,)
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

save_object(num_sensible_per_point, "29_July_Pos_Class_168000_173000/num_sensible_per_point.dat")
save_object(valid_smiles_final, "29_July_Pos_Class_168000_173000/valid_smiles.dat")
save_object(all_smiles_final, "29_July_Pos_Class_168000_173000/all_smiles.dat")

save_object(y_con_5, "29_July_Pos_Class_168000_173000/y_con_5.dat")
save_object(y_con_10, "29_July_Pos_Class_168000_173000/y_con_10.dat")
save_object(y_con_20, "29_July_Pos_Class_168000_173000/y_con_20.dat")
save_object(y_con_30, "29_July_Pos_Class_168000_173000/y_con_30.dat")
save_object(y_con_40, "29_July_Pos_Class_168000_173000/y_con_40.dat")
save_object(y_con_50, "29_July_Pos_Class_168000_173000/y_con_50.dat")
save_object(y_con_60, "29_July_Pos_Class_168000_173000/y_con_60.dat")
save_object(y_con_70, "29_July_Pos_Class_168000_173000/y_con_70.dat")
save_object(y_con_80, "29_July_Pos_Class_168000_173000/y_con_80.dat")
save_object(y_con_90, "29_July_Pos_Class_168000_173000/y_con_90.dat")

print(datetime.now() - startTime)
