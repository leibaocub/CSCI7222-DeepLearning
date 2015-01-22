#!/usr/bin/python 

import numpy as np
import math
from numpy import linalg as LA
from sklearn import linear_model
from neuralnetworkAux import *


##########
# main function
##########
def main():
# read data from filename file
# output format is a matrix
# 1st column is X
# 2nd column is y
# 3rd column is z - classification
    filename = "assign1_data.txt"
    data = np.loadtxt(filename,skiprows=1)
    m = data.shape[0]  ## size of w
    n = data.shape[1] ## size of w
    x = data[:,0:-2]
    y = data[:,[-2]] # size of y is m x 1
    z = data[:,[-1]] # size of z is m x 1

##### 
#part 1  
##### 
## find w by least square minimization
# treat bias term as additional dimension with value 1
    print('-' * 60)
    print "normal equation"
    X = np.insert(x,0,1,axis=1) 
    XT = X.T
    w_1 = np.linalg.solve(np.dot(XT, X),np.dot(XT, y))
    print w_1.flatten().tolist()

    clf = linear_model.LinearRegression()
    clf.fit(x,y)
    print "sklearn"
    print clf.intercept_, clf.coef_


##### 
#part 2
## linear regression
##### 
    print('-' * 60)
    print "linear regression"
    (w_2,_) = gradient_descent(X, y, alpha = 5e-3,  fun_eva= linear_regression,
                                maxiter = 200) 
    print "batch",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err

#### mini-batch learning
    (w_2,_) = gradient_descent(X, y, alpha = 5e-3, fun_eva = linear_regression,batchsize = 10,maxiter = 200) 
    print "mini_batch",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err

#### online learning
    (w_2,_) = gradient_descent(X, y, alpha = 5e-3, fun_eva = linear_regression, batchsize=1,maxiter=200) 
    print "online",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err
  
##### 
#part 3
## perceptron learning
##### 
### batch learning
    print('-' * 60)
    print "perceptron learning"
    (w_3,_) = gradient_descent(X, z, fun_eva = perceptron_learning) 
    print "batch",w_3.flatten().tolist()/w_3[0]
    res = classification(X,w_3)
    accu = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accu

#### mini-batch learning
    (w_3,_) = gradient_descent(X, z, fun_eva = perceptron_learning, batchsize = 10,
    maxiter = 200) 
    print "mini_batch",w_3.flatten().tolist()/w_3[0]
    res = classification(X,w_3)
    accu = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accu
#
#### online learning
    (w_3,_) = gradient_descent(X, z, fun_eva = perceptron_learning, batchsize=1,
    maxiter = 200) 
    print "online",w_3.flatten().tolist()/w_3[0]
    res = classification(X,w_3)
    accu = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accu

###### 
##part 4
### out-of-sample evaluation
### perceptron learning
###### 
    print('-' * 60)
    train_index = [25,50,75]
    test_set = X[-25:] # last 25 samples
    test_tar = z[-25:]
    accu_set = []
    for i in train_index:
        train_set = X[:i]
        train_tar = z[:i]
        (w,_) = gradient_descent(train_set,train_tar,fun_eva = perceptron_learning)
        res = classification(test_set,w)
        accu = (1.0 - sum(res != test_tar)/float(test_set.shape[0])) * 100.0
        accu_set.extend(accu)

    print train_index
    print accu_set

##########
# call main function when executing
##########
if __name__ == "__main__":
    main()



