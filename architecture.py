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


class CNN_A(nn.Module):


    def __init__(self):

        super(CNN_A,self).__init__()

        self.conv1=nn.Sequential(nn.Conv1d(in_channels=3,out_channels=64,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(32, 3, 1, 1)
        print(self.size)


        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)
        print(self.size)

        self.conv2=nn.Sequential(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        print(self.size)
        
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)
        print(self.size)

        self.conv3=nn.Sequential(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=(3,3), stride=(1,1),padding=1),
                nn.ReLU()
                )
        self.size=compute_out_size(self.size, 3, 1, 1)
        print(self.size)
        
        self.maxpool3=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.size=compute_out_size(self.size, 2, 0, 2)
        print(self.size)

    


        self.FC1=nn.Sequential(nn.Linear(self.size*self.size*256, 4096),
                    nn.ReLU()
                )

        self.FC2=nn.Sequential(nn.Linear(4096, 1000),
                    nn.ReLU()
                )

        self.FC3=nn.Linear(1000, 10)
        self.softmax=nn.Softmax(dim=1)





    def forward(self, inputs):

        out=self.conv1(inputs)
        out=self.maxpool1(out)
        out=self.conv2(out)
        out=self.maxpool2(out)
        out=self.conv3(out)
        out=self.maxpool3(out)

        out=out.reshape(-1, 4*4*256)

        out=self.FC1(out)
        out=self.FC2(out)
        out=self.FC3(out)

        out=self.softmax(out)

        return out



