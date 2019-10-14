import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_out_size(in_size, F, P, S):

    """
    Compute one-dimensionsal size after a convolutional or maxpooling layer

    Inputs:
        in_size             inputs one-dimensional size (before the layer)
        F                   filter kernel size
        P                   padding size
        S                   stride size

    Output :
        out_size            output one-dimensional size (after the layer)
    """

    out_size=int((in_size-F+2*P)/S)+1

    return out_size


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_C(nn.Module):


    def __init__(self):

        super(CNN_C,self).__init__()

        self.conv1=nn.Sequential(nn.Conv1d(in_channels=3,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(32, 3, 1, 1)     
        self.conv2=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)





        self.conv3=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv4=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)




        self.conv5=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv6=nn.Sequential(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool3=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)



        self.conv7=nn.Sequential(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv8=nn.Sequential(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool4=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)





        self.FC1=nn.Sequential(nn.Linear(2*2*256, 128),
                    nn.ReLU()
                )


        self.FC2=nn.Linear(128, 10)
        self.softmax=nn.Softmax(dim=1)





    def forward(self, inputs):

        out=self.conv1(inputs)
        out=self.conv2(out)
        out=self.maxpool1(out)
        
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.maxpool2(out)

        out=self.conv5(out)
        out=self.conv6(out)
        out=self.maxpool3(out)

        out=self.conv7(out)
        out=self.conv8(out)
        out=self.maxpool4(out)

        out=out.reshape(-1, 2*2*256)

        out=self.FC1(out)
        out=self.FC2(out)

        out=self.softmax(out)


        return out



class CNN_B(nn.Module):


    def __init__(self):

        super(CNN_B,self).__init__()

        self.conv1=nn.Sequential(nn.Conv1d(in_channels=3,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(32, 3, 1, 1)     
        self.conv2=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)





        self.conv3=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv4=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)




        self.conv5=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv6=nn.Sequential(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool3=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)







        self.FC1=nn.Sequential(nn.Linear(4*4*128, 128),
                    nn.ReLU()
                )


        self.FC2=nn.Linear(128, 10)
        self.softmax=nn.Softmax(dim=1)





    def forward(self, inputs):

        out=self.conv1(inputs)
        out=self.conv2(out)
        out=self.maxpool1(out)
        
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.maxpool2(out)

        out=self.conv5(out)
        out=self.conv6(out)
        out=self.maxpool3(out)

        out=out.reshape(-1, 4*4*128)

        out=self.FC1(out)
        out=self.FC2(out)

        out=self.softmax(out)

        return out



class CNN_A(nn.Module):


    def __init__(self):

        super(CNN_A,self).__init__()

        self.conv1=nn.Sequential(nn.Conv1d(in_channels=3,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(32, 3, 1, 1)     
        self.conv2=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=32,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)





        self.conv3=nn.Sequential(nn.Conv1d(in_channels=32,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        self.conv4=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)

        
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)



        self.FC1=nn.Sequential(nn.Linear(8*8*64, 128),
                    nn.ReLU()
                )


        self.FC2=nn.Linear(128, 10)
        self.softmax=nn.Softmax(dim=1)





    def forward(self, inputs):

        out=self.conv1(inputs)
        out=self.conv2(out)
        out=self.maxpool1(out)
        
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.maxpool2(out)

        out=out.reshape(-1, 8*8*64)

        out=self.FC1(out)
        out=self.FC2(out)

        out=self.softmax(out)

        return out

