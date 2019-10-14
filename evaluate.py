import torch
import torch.nn as nn
from data import testloader, batch_size_test
from architecture import CNN_A, CNN_B

PATH_model="./model/modelB.pt"


#Load the model
model=CNN_A()
model.load_state_dict(torch.load(PATH_model))
model.eval()


crossentropy=nn.CrossEntropyLoss()


Loss=0
Accuracy_Top1=0
Accuracy_Top5=0


for i, (inputs, labels) in enumerate(testloader, 0):

    outputs=model(inputs)

    batch_loss=crossentropy(outputs, labels)

    #Top 1 Accuracy
    _, predicted = torch.max(outputs, 1)
    correct= (predicted==labels).sum().item()

    batch_accuracy_top1=float(correct)/float(batch_size_test)


    #Top 5 Accuracy
    _,index_predicted=torch.topk(outputs,k=5, dim=1)

    correct_top5=0
    for p, index in enumerate(index_predicted,0):
        if labels[p] in index:
            correct_top5+=1

    
    batch_accuracy_top5=float(correct_top5)/float(batch_size_test)


    Loss+=batch_loss.item()
    Accuracy_Top1+=batch_accuracy_top1
    Accuracy_Top5+=batch_accuracy_top5


Loss=Loss/testloader.__len__()
Accuracy_Top1=Accuracy_Top1/testloader.__len__()
Accuracy_Top5=Accuracy_Top5/testloader.__len__()


print('TEST (10 000 samples) : Top1-Accuracy : %s  || Top5-Accuracy : %s  || Cross-Entropy : %s'%(round(Accuracy_Top1*100, 3), round(Accuracy_Top5*100,3), round(Loss, 2)))



