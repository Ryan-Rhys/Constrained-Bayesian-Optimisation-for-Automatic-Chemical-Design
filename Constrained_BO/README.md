# Constrained Bayesian Optimisation for Automatic Chemical Design using Variational Autoencoders

Welcome to the repo accompanying the paper "Constrained Bayesian Optimisation for Automatic Chemical Design usingg Variational Autoencoders"

https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc04026a

and

https://arxiv.org/abs/1709.05501

The code is based heavily on the implementation of the Aspuru-Guzik group:

https://github.com/aspuru-guzik-group/chemical_vae

# INSTALL

It is recommended that you install dependencies within a virtual environment. For example, using conda you would run,
from the Constrained_BO directory, the commands:

conda create -n env_name

source activate env_name

pip install -r requirements.txt

# USAGE

## Feature Generation

The scripts

generate_latent_features_and_targets_example.py
generate_qed_features_and_targets.py
generate_solo_qed.py

must be run first in order to create the features and targets for molecule generation.

## Toy Example

located in the **Branin_Hoo** folder.

Constrained Bayesian Optimisation on the toy Branin-Hoo function.

## Constrained Bayesian Optimization for Automatic Chemical Design

located in the **Chemical_Design** folder

The **Unconstrained** directory contains scripts that generate molecules using unconstrained Bayesian Optimisation.
The **Constrained** directory contains scripts that generate molecules using constrained Bayesian Optimisation.

Within these directories there are 3 scripts optimising the following objectives: 

a) bo_gp.py -> logP + SA + ring-penalty

b) bo_gp_qed.py -> QED + SA + ring-penalty

c) bo_gp_solo_qed.py -> QED

## Making Custom Training Data for Learning the Constraint Function

The script Constrained_BO/Chemical_Design/autoencoder/optimizers/make_training_data.py may be used to create training data for the constraint function according to different criterion such as the number of valid decodings required to satisfy the constraint.
