#!/usr/bin/python 
####################
## auxilary functions for neural network
####################

import numpy as np
import math
import random
from numpy import linalg as LA


####################
# 2015Spring: Neural Network and deep learning
# python script for part 1 of 
# assignment 1 
###############
# cost function and gradient
###############
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

def perceptron_learning(w,x,y):
    coef = 1.0
    dy = classification(x,w) - y
    #cost = coef * LA.norm(dy,ord=2)
    cost = coef * np.vdot(dy, dy)
    gradient = np.dot(x.T,dy)
    return (cost,gradient)
  

##########
# function evaluation rule
##########
def regression(x,w):
    return np.dot(x, w)

def classification(x,w):    
    res = np.dot(x, w) > 0.0
    return res


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

