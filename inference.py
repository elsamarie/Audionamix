import torch
from data import testloaderinference, classes
from architecture import CNN_A, CNN_B
import random
from plot_image import *
PATH_model="./model/modelAfinal.pt"

#Load the model 
model=CNN_A()
model.load_state_dict(torch.load(PATH_model))
model.eval()

#Get one random sample from the test dataset (testloaderinference)
n_samples=testloaderinference.__len__()
index=random.randint(0,n_samples)

#Getthe image and the class target associated to the sample
image, target = testloaderinference.dataset.__getitem__(index)

image=image.reshape(1,3,32,32)

#Compute the output of the model for the image input
output = model(image)

#Compute the predicted class 
_, target_predicted=torch.max(output, 1)


#Get the textual information of the predicted class and the target class
class_predicted=classes[target_predicted]
class_target=classes[target]

print('Class predicted : %s || Class target : %s'%(class_predicted, class_target))

#Visualize the image 
imshow(torchvision.utils.make_grid(image))
