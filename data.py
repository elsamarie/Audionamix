import torch
import torchvision
import torchvision.transforms as transforms

import torch.utils.data as data

fraction_val=0.3
fraction_train=1-fraction_val
batch_size=50
batch_size_val=500
batch_size_test=500


#Download the train (50000) and the test (10000) dataset from CIFAR10


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Image Normalization

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



#Creation of the train, validation and the test datasets for Pytorch : Dataloader for which the samples are batched


#Train dataset
trainsampler=data.RandomSampler(trainset, replacement=True, num_samples=int(fraction_train*5000))
trainloader = data.DataLoader(trainset, batch_size=batch_size, sampler=trainsampler , num_workers=2)

#Validation dataset
valsampler=data.RandomSampler(trainset, replacement=True, num_samples=int(fraction_val*5000))
valloader= data.DataLoader(trainset, batch_size=batch_size_val, sampler=valsampler, num_workers=2)


#Test dataset for which batch is one sample for the inference.py file
testloaderinference = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)


#Test dataset for model evaluation
testsampler=data.RandomSampler(testset, replacement=True, num_samples=1000)
testloader=data.DataLoader(testset, batch_size=batch_size_test, sampler=testsampler, num_workers=2)


#Text Classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



