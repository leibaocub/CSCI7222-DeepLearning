#!/usr/bin/python 
####################
## neural network
####################
import numpy as np
import math
import random
from funset import *
from numpy import linalg as LA
from functools import partial
####################
# 2015Spring: Neural Network and deep learning
###############
##########
# function evaluation rule
## return the maximum label, starting from 0
##########
def multiclass_indicator(x, w, fun = mysoftmax):
    xw = fun(x,w)[0]
    maxind = np.argmax(xw,axis=1)
    labels = np.ravel(maxind)
    return labels
###############
## wrapper for composite cost function and activation function
## from chain rule
###############
def myfun(cost_fun, active_fun, w, x, y):
    (z, gradz) = active_fun(x,w)

    (cost, gradcost) = cost_fun(y,z)
    
    gradtemp = gradcost * (gradz) # element-wise

    gradw = np.dot(x.T, gradtemp)

    return (cost, gradw)

###############
## wrapper for cross entropy and softmax,
## dC/dnet is just (z-y)
###############
def myfun_default(w, x, y):
    z = mysoftmax(x,w)[0]

    cost = cross_entropy(y,z)[0]
    
    gradtemp = z - y
    gradw = np.dot(x.T, gradtemp)

    return (cost, gradw)


###############
## wrapper for cross entropy and softmax,
## dC/dnet is just (z-y)
###############
###############
## initialize weight for neural network
###############
def initialw(m,n,epsilon=1.0):
    w = np.random.normal(0.0,epsilon,(m,n))
    l1 = LA.norm(w, ord=1, axis=0) ## row sum
    w = 2.0 * w/l1 ### normalize
    return w
    
#####
# gradient descent 
## support mini-batch learning.
### input
## x:               train-feature, np array, m x n, required
## y:               train-target, np array, m x 1, required
## alpha:           learning rate, default 1.0
## theta:           momentum method, default 0.0 
## maxiter:         max iteration number, default 1000
## tol:             error tolerance. error use squared error, default 1e-5
## fun_eva:         function to evaluate, required
##                  provide cost and gradient (general sense),
## initialguess:    initialguess for w, default 0
## batchsize:       size for mini-batch learning, default batch learning

### output (w,err_his)
## w:               parameter found. np array, m x 1
## err_his:         track of squared error along iterations
##########

####################
# gradient descent with train and test comparison
####################
def gradient_descent(x, y, alpha=1.0, theta = 0.0,
                maxiter = 1000, tol = 1e-5,
                fun_eva = None,
                initialguess = None, batchsize = None,
                xtest = None, ytest = None):
    if (initialguess == None):
    # by default, set initial guess to be all 0
        try:
            w = np.zeros((x.shape[1],y.shape[1]))
        except IndexError:
            # y should be (m,1) or (m,n)
            print "Error, the dimension of y should be 2! Try again..."
            raise
    else:
        w = initialguess

    if (xtest == None and ytest == None):
        debug = False
    else:
        debug = True

    tolnum = x.shape[0]
    indices = range(tolnum)
    if (batchsize == None):
    # batch learning, by default
        batchsize = tolnum
  
    # intervals to split the training set 
    intervals = range(0,tolnum,batchsize)
    intervals.append(tolnum)
    
    err_his = []
    err_his_test = []

    deltaw = 0.0
    for i in xrange(maxiter):
    ## get the random shuffle of the indices
        random.shuffle(indices)

        for j in xrange(1,len(intervals)):
            js = intervals[j-1]
            je = intervals[j]
            ix = indices[js:je]
            (cost,grad) = fun_eva(w,x[ix],y[ix])
                
            deltaw = theta * deltaw - (1.0 - theta) * alpha * grad
            w = w + deltaw

        cost = fun_eva(w,x,y)[0]
        err_his.append(cost)

        if (debug):
            costtest = fun_eva(w,xtest,ytest)[0]
            err_his_test.append(costtest)

        if ( cost < tol):
            break## happy w/ cost 

    return (w,err_his,err_his_test)

    
####################
# logistic units
####################
def logistic_neurons(x,y,alpha=1.0, theta = 0.0, maxiter = 1000, 
                tol = 1e-5, 
                cost_fun = cross_entropy,
                active_fun = mysoftmax,
                initialguess = None, 
                batchsize = None,
                xtest = None, ytest = None):


    if (initialguess == None ):#None
        try:
            initialguess = initialw(x.shape[1],y.shape[1])
        except IndexError:
            # y should be (m,1) or (m,n)
            print "Error, the dimension of y should be 2! Try again..."
            raise

    if (cost_fun == cross_entropy and active_fun == mysoftmax):
        fun_eva = myfun_default ## gradient computed algebraically
    else:
        ## compute gradient from chain rule
        fun_eva = partial(myfun, cost_fun, active_fun)

    w, errhis, errhis_test = gradient_descent(x,y, 
                        alpha, theta, maxiter, 
                        tol, fun_eva, initialguess, batchsize,
                        xtest, ytest)

    yvar = np.mean(y, axis=0)
    yvarsum = np.sum((y - yvar) * (y - yvar))
    errhis = errhis/(yvarsum)
    
    if (xtest != None):
        yvar = np.mean(ytest, axis=0)
        yvarsum = np.sum((ytest - yvar) * (ytest - yvar))
        errhis_test = errhis_test/(yvarsum)

    return (w, errhis, errhis_test)


    
