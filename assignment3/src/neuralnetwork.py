#!/usr/bin/python 
####################
## neural network
####################

import numpy as np
import math
import random
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from functools import partial
####################
# 2015Spring: Neural Network and deep learning
###############
###############
## square error as cost function
## mytanh as activation function
###############
def square_error(t,y):
    delta = t - y
    cost = 0.5 * np.vdot(delta,delta)
    grad = - delta ## dC/dy
    return (cost, grad)

def mytanh(x,w):
    net = np.dot(x,w)
    y = 2.0/(1.0 + np.exp(-net)) - 1.0
    grad = 0.5 * (1.0 + y) * (1.0 - y) ## element-wise
    return (y, grad)

def mysigmoid(x,w):
    net = np.dot(x,w)
    y = 1.0/(1.0 + np.exp(-net))
    grad = y * (1.0 - y) ## element-wise
    return (y, grad)
###############
## cross entropy as cost function
## softmax as activation function
###############
def cross_entropy(t,y):
    pr = np.log(y)
    cost = -np.vdot(t,pr)
## maybe need to compute derivative too
    grad = - t/y ## elementwise
    return (cost, grad)

######
## mysoftmax can only be output fun right now!!!
def mysoftmax(x,w):
    net = np.dot(x,w)
    y = np.exp(net)
    sumy = np.sum(y,axis=1)
    y = y/sumy[:, np.newaxis]
## maybe need to compute derivative too
    grad = y - y * y ## only for the diagonal part
    return (y,grad)

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

def myfun_fdz(cost_fun, active_fun, w, x, y):
    (z, gradz) = active_fun(x,w)

    (cost, gradcost) = cost_fun(y,z)
    
    gradz = gradcost * (gradz) # element-wise

    return (cost, gradz)
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


def myfun_default_fdz(w, x, y):
    z = mysoftmax(x,w)[0]

    cost = cross_entropy(y,z)
    
    gradz = z - y

    return (cost, gradz)
###############
## wrapper for cross entropy and softmax,
## dC/dnet is just (z-y)
###############
###############
## initialize weight for neural network
###############
def nn_initial(m,n,epsilon=1.0):
    w = np.random.normal(0.0,epsilon,(m,n))
    l1 = LA.norm(w, ord=1, axis=0) ## row sum
    w = 2.0 * w/l1 ### normalize
    return w

def nn_labely(y, outputfun):
    tolnum = y.size
    uy = np.unique(y)
    uynum = uy.size
    if (outputfun == mysigmoid):
##label y as 0 or 1
        yy = np.zeros((tolnum,uynum))
        for i in xrange(tolnum):
            yy[i,y[i]] = 1.0
    elif (outputfun == mytanh):
##label y as -1 or 1
        yy = np.empty((tolnum,uynum))
        yy.fill(-1.0)
        for i in xrange(tolnum):
            yy[i,y[i]] = 1.0
        
    else:
##label y as 0 or 1
        yy = np.zeros((tolnum,uynum))
        for i in xrange(tolnum):
            yy[i,y[i]] = 1.0

        yy = np.exp(yy)
        sumy = np.sum(yy,axis=1)
        yy = yy/sumy[:, np.newaxis]

    return yy
    
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
            initialguess = nn_initial(x.shape[1],y.shape[1])
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


####################
# neural network
####################
def neural_network(x, yy, alpha=1.0, theta = 0.0, maxiter = 1000, 
                tol = 1e-5, 
                cost_fun = cross_entropy,
                output_fun = mysigmoid,
                hidden_fun = mysigmoid,
                hidden_units = [8],
                batchsize = None):
    
##########
## label y according to the output function
##########
    y = nn_labely(yy,output_fun)
    try:
        myunits = [x.shape[1]] + hidden_units + [y.shape[1]]
    except IndexError:
        # y should be (m,1) or (m,n)
        print "Error, the dimension of y should be 2! Try again..."
        raise


##########
## initialize weights    
##########
    numl= len(myunits) - 1
    layers = map(lambda i: {}, xrange(0,numl)) 
    ###hidden layers
    for i in xrange(numl):
        layers[i]['w'] = nn_initial(myunits[i]+1, myunits[i + 1], 1.0)

##########
## get fun wrapper for output layer
##########
    if (cost_fun == cross_entropy and output_fun == mysoftmax):
        hidden2output = myfun_default_fdz ## gradient computed algebraically
    else:
        ## compute gradient from chain rule
        hidden2output = partial(myfun_fdz, cost_fun, output_fun)

##########
## generate subsets for mini-batch training
##########
    tolnum = x.shape[0]
    indices = range(tolnum)
    if (batchsize == None):
    # batch learning, by default
        batchsize = tolnum
  
    # intervals to split the training set 
    intervals = range(0,tolnum,batchsize)
    intervals.append(tolnum)
    
    errhis = []

    for itern in xrange(maxiter):
    ## get the random shuffle of the indices
        random.shuffle(indices)

        for j in xrange(1,len(intervals)):
            js = intervals[j-1]
            je = intervals[j]
            ix = indices[js:je]

            train_network(layers, x[ix], y[ix], hidden_fun, hidden2output, alpha, theta)

#####evaluate the value of cost function
        yhidden = x ## dummy
        for li in xrange(numl - 1):
            a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
            yhidden = hidden_fun(a,layers[li]['w'])[0]
        
        a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
        cost = hidden2output(layers[-1]['w'], a ,y)[0]

        errhis.append(cost)
#####

        if ( cost < tol):
            break## happy w/ cost 

    if (cost_fun == square_error):
        yvar = np.mean(y, axis=0)
        yvarsum = np.sum((y - yvar) * (y - yvar))
        errhis = errhis/(yvarsum)
    else:
        errhis = [i/float(x.shape[0]) for i in errhis]

    w = [d['w'] for d in layers]
    return (w, errhis)

def train_network(layers, xin, yin, hidden_fun, hidden2output, alpha, theta):
    ### forward feeding
    numhid = len(layers) - 1
    yhidden = xin
    for li in xrange(numhid):
        a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
        layer = layers[li]
        layer['a'] = a

        w = layer['w']
        yhidden, dfdz = hidden_fun(a , w)
        layer['dfdz'] = dfdz

    ### output layer, layers[-1]
    a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
    layer = layers[-1]
    layer['a'] = a
    w = layer['w']
    cost, gradz = hidden2output(w,a,yin)
    layer['gradz'] = gradz
    
    ### backward propogation
    for li in reversed(xrange(numhid)):
        fz = layers[li]['dfdz']
        w = layers[li+1]['w']
        gradz = layers[li+1]['gradz']
        grada = np.dot(gradz, w[1:].T) ## ignor bias term
        ### for next layer
        layers[li]['gradz'] = grada * fz

    ### forward updating weights
    for layer in layers:
        a = layer['a']
        gradz = layer['gradz']

        gradw = np.dot(a.T, gradz) ## elementwise

        try:
            layer['deltaw'] =  theta * layer['deltaw'] - (1.0 - theta) * alpha * gradw
        except KeyError:
            layer['deltaw'] =  - alpha * gradw

        layer['w'] = layer['w'] + layer['deltaw']


def nn_predict(w,x,output_fun = mysoftmax,hidden_fun = mysigmoid):
    numl = len(w)
    yhidden = x
    for i in xrange(numl - 1):
        a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
        yhidden = hidden_fun(a,w[i])[0]

    a = np.insert(yhidden,0,1,axis=1) # add 1 at the fist column to handle bias term
    outputs = output_fun(a, w[-1])[0]
    maxind = np.argmax(outputs,axis=1)
    labels = np.ravel(maxind)
    
    return labels
    
