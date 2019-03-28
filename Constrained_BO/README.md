Welcome to the code accompanying the paper "Constrained Bayesian Optimisation for Automatic Chemical Design"

https://arxiv.org/abs/1709.05501

The code is based heavily on the implementation of the Aspuru-Guzik group:

https://github.com/aspuru-guzik-group/chemical_vae

INSTALL

It is recommended that you install dependencies within a virtual environment. For example, using conda you would run,
from the Constrained_BO directory, the commands:

conda create -n env_name

source activate env_name

pip install -r requirements.txt

USAGE

The scripts

generate_latent_features_and_targets_example.py
generate_qed_features_and_targets.py
generate_solo_qed.py

must be run first in order to create the features and targets for molecule generation.

1) Branin_Hoo

Constrained Bayesian Optimisation on the toy Branin-Hoo function.

2) Chemical_Design

The Unconstrained directory contains scripts that generate molecules using unconstrained Bayesian Optimisation.
The Constrained directory contains scripts that generate molecules using constrained Bayesian Optimisation.

Within these directories there are 3 scripts optimising the following objectives: 

a) bo_gp.py -> logP + SA + ring-penalty

b) bo_gp_qed -> QED + SA + ring-penalty

c) bo_gp_solo_qed -> QED

The Initialisation directory contains code to generate training data for the binary classification neural network in 
the scripts Pos_Gen.py and Neg_Gen.py.
