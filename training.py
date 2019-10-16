# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from architecture import CNN_A, CNN_B, CNN_C
from data import *
from plot import *

#Parameters
learning_rate=0.001
momentum=0.9
n_epoch=50
PATH_model="./model/modelAfinal.pt"


#Initialisation 
print(PATH_model)
model=CNN_A()


#Get number of parameters of the model
number_parameter=0
tensor_list = list(model.state_dict().items())
for layer_tensor_name, tensor in tensor_list:
    print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
    number_parameter+=torch.numel(tensor)

print('total amount of parameters : {}'.format(number_parameter))

#Adam optimizer (used for the Loss backpropagation)
optimizer=optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

#Loss object : CrossEntropy
crossentropy=nn.CrossEntropyLoss()


#Performances tracker
train_loss=np.array([])
train_accuracy=np.array([])

val_loss=np.array([])
val_accuracy=np.array([])




print('Beginning of Training')

for n in range(0, n_epoch):

    #Performance trackers for one epoch
    epoch_loss=0
    epoch_accuracy=0

    bar=ProgressBar(trainloader.__len__(),'Epoch[{}/{}]:'.format(n+1, n_epoch), ' ',50)
    bar.initProgressBar()

    
    for i, (inputs, labels) in enumerate(trainloader, 0): #Train the model : forward and backward


        optimizer.zero_grad()
        outputs=model(inputs)

        #Crossentropy loss
        batch_loss=crossentropy(outputs, labels)

        #BackPropagation of the loss to update the weights of the model
        batch_loss.backward()
        optimizer.step()

        #Accuracy
        _, predicted = torch.max(outputs, 1)
        correct= (predicted==labels).sum().item()

        batch_accuracy=float(correct)/float(batch_size)



        #Update the performance trackers
        epoch_accuracy+=batch_accuracy
        epoch_loss+=batch_loss.item()

        bar.loopProgressBar(i)


    epoch_accuracy=epoch_accuracy/trainloader.__len__()
    epoch_loss=epoch_loss/trainloader.__len__()

    print("Training set   : Loss : %s || Accuracy : %s "%(round(epoch_loss,3), round(epoch_accuracy*100,3)))

    #Save the performance for the training set
    train_loss=np.append(train_loss, epoch_loss)
    train_accuracy=np.append(train_accuracy, epoch_accuracy)




    #Performance trackers for one epoch
    epoch_loss=0
    epoch_accuracy=0



    for i, (inputs, labels) in enumerate(valloader, 0): #Validate the model : only forward

        outputs=model(inputs)

        #Crossentropy Loss
        batch_loss=crossentropy(outputs, labels)

        #Accuracy
        _, predicted = torch.max(outputs, 1)
        correct= (predicted==labels).sum().item()

        batch_accuracy=float(correct)/float(batch_size_val)

        epoch_accuracy+=batch_accuracy
        epoch_loss+=batch_loss.item()




    epoch_accuracy=epoch_accuracy/valloader.__len__()
    epoch_loss=epoch_loss/valloader.__len__()

    print("Validation set : Loss : %s || Accuracy : %s "%(round(epoch_loss,3), round(epoch_accuracy*100,3)))

    #Save the performance for the validation set
    val_loss=np.append(val_loss, epoch_loss)
    val_accuracy=np.append(val_accuracy, epoch_accuracy)






#Plot the performance trackers
plot_crossentropy_loss(train_loss, val_loss,n_epoch)
plot_accuracy(train_accuracy, val_accuracy,n_epoch)


#Save the model
torch.save(model.state_dict(), PATH_model)

print('End of Training')