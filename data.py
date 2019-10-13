import torch
import torchvision
import torchvision.transforms as transforms

import torch.utils.data as data

fraction_val=0.3
fraction_train=1-fraction_val
batch_size=30


#Download the train (50000) and the test (10000) dataset from CIFAR10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



#Creation of the train, validation and the test dataset for Pytorch


trainsampler=data.RandomSampler(trainset, replacement=True, num_samples=int(fraction_train*(trainset.__len__())))
valsampler=data.RandomSampler(trainset, replacement=True, num_samples=int(fraction_val*(trainset.__len__())))

trainloader = data.DataLoader(trainset, batch_size=batch_size, sampler=trainsampler , num_workers=2)
print(trainloader.__len__())
valloader= data.DataLoader(trainset, batch_size=4, sampler=valsampler, num_workers=1)


testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



