#!/usr/bin/python 
import re
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

####################
## read file
####################
def readFile(filename):
    y = []
    features = []
    with open(filename) as f:
        s = []
        for line in f:
            if (re.search('[.]',line) != None):# found float
                s += map(float,line.split())
            elif (re.search('-',line) != None):
                ss = re.findall("[a-z]+|\d+",line) # split line into ["train","0","1"]
                y.append(int(ss[1]))
            else:
                features.append(s)
                s = []

    y = np.array(y)
    x = np.array(features)
    return (x,y)

####################
## compute accuracy
####################
def compute_accuracy(pred,truesol):
#### check shape of predictions and true solutions    
    if ( not np.array_equiv(pred.shape, truesol.shape)):
        print "Error: predictions and true solutions have different shape!"
        exit(1)
    
    if (pred.ndim == 1):#vector, shape (n,)
        acc = ( 1.0 - sum(pred !=  truesol )/ float(pred.shape[0])) 
        
    elif (pred.shape[1] == 1): #vector, shape(n,1)
        acc = ( 1.0 - sum(pred[:,0] !=  truesol[:,0] )/ float(pred.shape[0])) 
        
    else:#prev is a matrix
        count = 0
        for i,row in enumerate(pred):
            if np.array_equiv(row, truesol[i]):
                count += 1

        acc = count/ float(pred.shape[0])
    return acc * 100.0


####################
#### plot_confusionmatrix
## ytest: true labels
## ypred: predicted labels
## alphabet: list of strings for labels. If none, use 0,1,2,...
## filename: output file for plot of confusion matrix
####################
def plot_confusionmatrix(ytest,ypred,
                        alphabet = None,
                        filename = None):
    conf_arr = confusion_matrix(ytest, ypred)
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white')

    cb = fig.colorbar(res)
#alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#    alphabet = '1234567'
####
# labels for ploting not given, use 0 - height. 
#    height: # of unique labels
####
    if (alphabet == None):
        alphabet = ["%d" %i for i in xrange(height)]

    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')

    if (filename == None):
        plt.show()
    else:
        plt.savefig(filename)
    
####################

