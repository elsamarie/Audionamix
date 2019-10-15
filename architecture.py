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


"""
CNN C : 
4 conv. unit

conv1 3x3 32 RELU
conv2 3x3 32 RELU
maxpool1 2x2

conv3 3x3 64 RELU
conv4 3x3 64 RELU
maxpool2 2x2

conv5 3x3 128 RELU
conv6 3x3 128 RELU
maxpool3 2x2

conv7 3x3 256 RELU
conv8 3x3 256 RELU
maxpool4 2x2

FC1 1024 128 RELU
FC2 128 10 Softmax


"""


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


"""
CNN B : 
3 conv. unit

conv1 3x3 32 RELU
conv2 3x3 32 RELU
maxpool1 2x2

conv3 3x3 64 RELU
conv4 3x3 64 RELU
maxpool2 2x2

conv5 3x3 128 RELU
conv6 3x3 128 RELU
maxpool3 2x2


FC1 2048 128 RELU
FC2 128 10 Softmax


"""

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

"""
CNN A : 
2 conv. unit

conv1 3x3 32 RELU
conv2 3x3 32 RELU
maxpool1 2x2

conv3 3x3 64 RELU
conv4 3x3 64 RELU
maxpool2 2x2

FC1 4096 128 RELU
FC2 128 10 Softmax


"""

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

