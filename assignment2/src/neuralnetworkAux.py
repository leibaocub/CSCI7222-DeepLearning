#!/usr/bin/python 
####################
## auxilary functions for neural network
####################

import numpy as np
import math
import random
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt

####################


####################
# 2015Spring: Neural Network and deep learning
# python script for part 1 of 
# assignment 1 
###############
# cost function and gradient
###############
##########
# function evaluation rule
##########
def regression(x,w):
    return np.dot(x, w)

def classification(x,w):    
    res = np.dot(x, w) > 0.0
    return res

def multiclass_classification(x,w):
    xw = np.dot(x,w)
    maxind = np.argmax(xw,axis=1)
    labels = np.zeros(xw.shape)
    for i,ind in enumerate(maxind):
        labels[i,ind] = 1.0
    return labels

##########
## least square minimization
## y = x^T * w + b
## Y = Xw
## where Y = (y1,y2,...,yn)
## X = (x1^T; x2^T,...)
## xi = (1,xi)
##########
def linear_regression(w,x,y):
    coef = 1.0
    dy = regression(x,w) - y
# choose cost function as ||Xw - y||^2 * coef
    cost = coef * np.vdot(dy,dy) 
    gradient = 2.0 * coef * np.dot(x.T,dy)
    return (cost,gradient)

def perceptron_learning(w,x,y,fun=classification):
    coef = 1.0
    dy = fun(x,w) - y
    #cost = coef * LA.norm(dy,ord=2)
    cost = coef * np.vdot(dy, dy)
    gradient = np.dot(x.T,dy)
    return (cost,gradient)
  
def multiclass_perceptron(w,x,y):
    return perceptron_learning(w,x,y,fun=multiclass_classification)

###############
#####
# gradient descent 
## support mini-batch learning.
### input
## x:               train-feature, np array, m x n, required
## y:               train-target, np array, m x 1, required
## alpha:           learning rate, default 1.0
## maxiter:         max iteration number, default 1000
## tol:             error tolerance. error use squared error, default 1e-5
## fun_eva:         function to evaluate. 
##                  provide cost and gradient (general sense),
##                  default linear_regression
## initialguess:    initialguess for w, default 0
## batchsize:       size for mini-batch learning, default batch learning

### output (w,err_his)
## w:               parameter found. np array, m x 1
## err_his:         track of squared error along iterations
##########
def gradient_descent(x,y,alpha=1.0, maxiter = 1000, 
                tol = 1e-5, 
                fun_eva = linear_regression,
                initialguess = None, batchsize = None):

    if (initialguess == None):
    # by default, set initial guess to be all 0
        w = np.zeros((x.shape[1],1))
    else:
        w = initialguess

    tolnum = x.shape[0]
    indices = range(tolnum)
    if (batchsize == None):
    # batch learning, by default
        batchsize = tolnum
  
    # intervals to split the training set 
    intervals = range(0,tolnum,batchsize)
    intervals.append(tolnum)
    
    err_his = []
    for i in xrange(maxiter):
    ## get the random shuffle of the indices
        random.shuffle(indices)

        for j in xrange(1,len(intervals)):
            js = intervals[j-1]
            je = intervals[j]
            ix = indices[js:je]
            (cost,grad) = fun_eva(w,x[ix],y[ix])
                
            w = w - alpha * grad 

        (cost,grad) = fun_eva(w,x,y)
        err_his.append(cost)

        if (cost < tol):
            break
  
    return (w,err_his)

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
def plot_confusionmatrix(ytest,ypred,alphabet, filename = 'confusion_matrix.png'):
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
                        verticalalignment='center')

    cb = fig.colorbar(res)
#alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#    alphabet = '1234567'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.savefig(filename)
    
####################
