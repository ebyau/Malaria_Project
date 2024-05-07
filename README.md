# Malaria_Project

* vig.py  
 This file provides an implementation of the ViG model, a Deep Graph Convolutional Network for image classification. It defines the model architecture, including the stem, backbone, and prediction layers, and provides different configurations for the model size.

* train.py  
Training script for training image classification models, particularly the ViG model. 

* data/myloader.py  
The create_loader function creates data loaders with various options for data augmentation, prefetching, and multi-epoch loading. It is used together with the training scripts to prepare the data for feeding into the model during training and evaluation.

* data/rasampler.py  
  The purpose of this sampler is to provide repeated augmentation during training
