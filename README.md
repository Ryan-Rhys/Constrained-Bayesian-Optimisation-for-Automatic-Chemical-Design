# Constrained Bayesian Optimisation for Automatic Chemical Design using Variational Autoencoders

Welcome to the code accompanying the paper "Constrained Bayesian Optimisation for Automatic Chemical Design"

https://arxiv.org/abs/1709.05501

The code is based heavily on the implementation of the Aspuru-Guzik group:

https://github.com/aspuru-guzik-group/chemical_vae

## INSTALL

Append the package directory location to your PYTHONPATH e.g. by editing the .bashrc file as follows:

```vim ~/.bashrc```

and adding

```PYTHONPATH="${PYTHONPATH}:/Users/path_to_directory/Constrained-Bayesian-Optimisation_for_Automatic_Chemical_Design"
export PYTHONPATH

source ~/.bashrc
```

It is recommended that you install dependencies within a virtual environment. For example, using conda you would run,
from the Constrained_BO_package directory, the commands:

```conda config --add channels conda-forge```

(to add conda-forge to existing channels)

```conda create -n env_name --file package-list.txt

source activate env_name

conda install rdkit==2017.09.3

cd Theano-master

python setup.py install

cd ..

conda install numpy==1.13.0

pip install git+https://github.com/rgbombarelli/keras.git#egg=Keras

pip install git+https://github.com/rgbombarelli/seya.git#egg=seya

pip install git+https://github.com/HIPS/autograd.git#egg=autograd
```

## USAGE

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

