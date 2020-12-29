
# About

This repository contains a PyTorch implementation of the extensions for The Grouploss for Deep Metric Learning https://arxiv.org/abs/1912.00385 that I developed in my [Master Thesis](https://drive.google.com/file/d/1kn1Lo-syhMLUbAzcnwQYgBwy9kxu_BYj/view?usp=sharing)



# PyTorch version

We use torch 1.1 and torchvision 0.2. While the training and inference should be able to be done correctly with the newer versions of the libraries, be aware that at times the network trained with torch > 1.2 might diverge or reach lower results.

We also support half-precision training via Nvidia Apex. 

# Reproducing Results

Extending the paper we support training in 4 datasets: CUB-200-2011, CARS 196, Stanford Online Products and In-shop Clothes Retrieval. Simply provide the path to the dataset in train.py and declare what dataset you want to use for the training. Training on some other dataset should be straightforwars as long as you structure the dataset in the same way as those four datasets.

The majority of experiments are done in inception with batch normalization. We provide support for the entire family of resnet and densenets. Simply define the type of the network you want to use in train.py.

In order to train and test the network run file train.py
