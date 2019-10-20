# Technical Test Audionamix

Convolutional Neural Networks for CIFAR-10 Classification based on [1]

[1] "Very Deep Convolutional Networks for Large-Scale Image Recognition”, Karen Simonyan and Andrew Zisserman, 2014

## Python Environment

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

mkdir ./data


## Dataset (CIFAR-10)

The CIFAR-10 is a labeled RGB 32x32 images dataset. It is divided into 50000 training images and 10000 test images.

The dataset is provided directly by tensorvision. For the first launch, the dataset will be download into the ./data folder. 


## RUNNING

By default, the available model is the CNN A saved in the ./model folder.

Train a model : python training.py

Note : Train a model with the entire training dataset could take 6 hours. 

Evaluate a model : python evaluate.py

Get the inference for one test sample : python inference.py


### Files 

I am used to build small python files, one for each step of the work. 

- data.py : Download or load the training, validation and the test dataloader 

- architecture.py : The three studied configurations of CNN 

- training.py : Train a model 

- plot.py : Plot the train and validation loss and accuracy over the learning iterations 

- evaluate.py : Evaluate a model with the test dataloader 

- inference.py : Get the inference class for one test sample 

- plot_image.py : Show a image of the dataset 


models are saved inside the ./model folder and figures inside the ./figure folder. 










