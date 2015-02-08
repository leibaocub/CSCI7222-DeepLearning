#!/usr/bin/python 

import numpy as np
import math
import random
from numpy import linalg as LA
from functools import partial
###############
## square error as cost function
## mytanh as activation function
###############
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

###############
## cross entropy as cost function
## softmax as activation function
###############
def square_error(t,y):
    delta = t - y
    cost = 0.5 * np.vdot(delta,delta)
    grad = - delta ## dC/dy
    return (cost, grad)

def cross_entropy(t,y):
    pr = np.log(y)
    cost = -np.vdot(t,pr)
## maybe need to compute derivative too
    grad = - t/y ## elementwise
    return (cost, grad)

