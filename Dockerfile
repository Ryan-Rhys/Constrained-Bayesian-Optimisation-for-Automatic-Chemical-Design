# Base image:
FROM python:2.7.0

MAINTAINER Ryan-Rhys Griffiths

RUN git clone https://github.com/Ryan-Rhys/Constrained-Bayesian-Optimisation-for-Automatic-Chemical-Design.git

RUN conda install rdkit==2017.09.03

RUN cd theano-master

RUN python setup.py install

RUN cd ..

RUN conda install numpy==1.13.0

RUN pip install git+https://github.com/rgbombarelli/keras.git#egg=Keras

RUN pip install git+https://github.com/rgbombarelli/seya.git#egg=seya

RUN pip install git+https://github.com/HIPS/autograd.git#egg=autograd
