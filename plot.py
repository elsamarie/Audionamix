# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
import sys, os

def plot_crossentropy_loss(train_loss, val_loss,n_epoch):

    epochs_axis=np.arange(0, n_epoch, step=1)

    plt.plot(epochs_axis, train_loss, label="train loss")
    plt.plot(epochs_axis, val_loss, label="valid loss")

    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy Loss')

    plt.title('Train and Valid CrossEntropy Loss')

    plt.legend()
    plt.savefig('./figure/Train_Valid_CrossEntropyLoss_graph.png')
    plt.show()


def plot_accuracy(train_accuracy, val_accuracy,n_epoch):

    epochs_axis=np.arange(0, n_epoch, step=1)

    plt.plot(epochs_axis, train_accuracy, label="train accuracy")
    plt.plot(epochs_axis, val_accuracy, label="valid accuracy")

    plt.xlabel('Epochs')
    plt.ylabel('Top-1 Accuracy (10 classes)')

    plt.title('Train and Valid Top 1 Accuracy')

    plt.legend()
    plt.savefig('./figure/Train_Valid_Accuracy_graph.png')
    plt.show()




class ProgressBar:
    
    def __init__(self, tot_iter, prefix, suffix, length):
        self.tot_iter = tot_iter
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
    
    def initProgressBar(self):
        self.printProgressBar(0, self.tot_iter, self.prefix, self.suffix, length = self.length)
    
    def loopProgressBar(self, ind):
        self.printProgressBar(ind + 1, self.tot_iter, self.prefix, self.suffix, length = self.length)
    
    def changeProgressBar(self, new_tot_iter, new_prefix, new_suffix, new_length):
        self.tot_iter = new_tot_iter
        self.prefix = new_prefix
        self.suffix = new_suffix
        self.length = new_length
    
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 75, fill = '█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        
        line='\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Print New Line on Complete
        if (iteration == total): 
            sys.stdout.write("\n")