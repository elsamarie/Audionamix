import torch
from data import testloaderinference, classes
from architecture import CNN_A, CNN_B
import random
from plot_image import *
PATH_model="./model/modelB.pt"


#Load the model
model=CNN_A()
model.load_state_dict(torch.load(PATH_model))
model.eval()


n_samples=testloaderinference.__len__()
index=random.randint(0,n_samples)


image, target = testloaderinference.dataset.__getitem__(index)

image=image.reshape(1,3,32,32)

output = model(image)

_, target_predicted=torch.max(output, 1)

class_predicted=classes[target_predicted]
class_target=classes[target]

print('Class predicted : %s || Class target : %s'%(class_predicted, class_target))

imshow(torchvision.utils.make_grid(image))
