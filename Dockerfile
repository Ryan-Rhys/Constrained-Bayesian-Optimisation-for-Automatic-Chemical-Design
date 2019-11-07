# Base image:
FROM continuumio/miniconda2

MAINTAINER Ryan-Rhys Griffiths

RUN conda config --add channels conda-forge

RUN conda install rdkit==2017.09.03

RUN conda install numpy==1.13.0

RUN pip install git+https://github.com/rgbombarelli/keras.git#egg=Keras

RUN pip install git+https://github.com/rgbombarelli/seya.git#egg=seya

RUN pip install git+https://github.com/HIPS/autograd.git#egg=autograd
