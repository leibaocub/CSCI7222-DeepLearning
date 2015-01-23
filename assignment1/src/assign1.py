#!/usr/bin/python 

import numpy as np
import math
from numpy import linalg as LA
from sklearn import linear_model
from neuralnetworkAux import *
import matplotlib.pyplot as plt


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
    (w_2,errgd) = gradient_descent(X, y, alpha = 5e-3,  fun_eva= linear_regression,
                                maxiter = 200) 
    print "batch",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err

#### mini-batch learning
    (w_2,errsgd) = gradient_descent(X, y, alpha = 5e-3, fun_eva = linear_regression,batchsize = 10,maxiter = 200) 
    print "mini_batch",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err

#### online learning
    (w_2,erronline) = gradient_descent(X, y, alpha = 5e-3, fun_eva = linear_regression, batchsize=1,maxiter=200) 
    print "online",w_2.flatten().tolist()

    res = regression(X,w_2)
    err = np.vdot(res - y,res-y)
    print err

###plot error history
    fig1 = plt.figure(1)
    iters = np.arange(1,len(errgd)+1)
    plt.plot(iters, errgd, iters, errsgd, iters, erronline)
    plt.legend(["batch","minibatch","online"])
    plt.title("Error History of different updating rules of LMS")
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    fig1.savefig("lms.eps")

###plot of batch sizes
    fig2 = plt.figure(2)
    for i in xrange(2,11,2):
        (_,err) = gradient_descent(X, y, alpha = 5e-3, fun_eva = linear_regression, batchsize=i,maxiter=100) 
        iters = np.arange(1,len(err)+1)
        plt.plot(iters, err,label = ("batchsize = %i " %i))
    
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    plt.legend()
    plt.title("Error History of mini-batch learning with different batchsize")
    fig2.savefig("batchsize_lms.eps")

  
##### 
#part 3
## perceptron learning
##### 
### batch learning
    print('-' * 60)
    print "perceptron learning"
    (w3gd,errgd) = gradient_descent(X, z, fun_eva = perceptron_learning,
                                    maxiter = 200) 
    print "batch",w3gd.flatten().tolist()/w3gd[0]
    res = classification(X,w3gd)
    accugd = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accugd

#### mini-batch learning
    (w3sgd,errsgd) = gradient_descent(X, z, fun_eva = perceptron_learning, batchsize = 10,
    maxiter = 200) 
    print "mini_batch",w3sgd.flatten().tolist()/w3sgd[0]
    res = classification(X,w3sgd)
    accusgd = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accusgd
#
#### online learning
    (w3online,erronline) = gradient_descent(X, z, fun_eva = perceptron_learning, batchsize=1,
    maxiter = 200) 
    print "online",w3online.flatten().tolist()/w3online[0]
    res = classification(X,w3online)
    accuonline = (1.0 - sum(res != z)/float(z.shape[0])) * 100.0
    print accuonline

###plot error history
    fig3 = plt.figure(3)
    iters = np.arange(1,len(errgd)+1)
    plt.plot(iters, errgd)
    plt.plot(iters, errsgd)
    plt.plot(iters, erronline)
    plt.legend(["batch","minibatch","online"])
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    plt.title("Error History of different updating rules for Perceptron Learning")
    fig3.savefig("errhis_perceptron.eps")

### plot
    colors = ['r' if t == 0 else 'b' for t in z]
    fig4 = plt.figure(4)
    plt.scatter(x[:,0], x[:,1], marker = 'o', c = colors)
## show plot
    scaledx = np.linspace(min(x[:,0]), max(x[:,0]),x.shape[0])
##batch
    plt.plot(scaledx, - w3gd[0]/w3gd[2] - w3gd[1]/w3gd[2] * scaledx,
    color = 'b', label = 'batch(%.0f%%)' %accugd)
##mini-batch
    plt.plot(scaledx, - w3sgd[0]/w3sgd[2] - w3sgd[1]/w3sgd[2] * scaledx,
    color = 'r', label = 'minibatch(%.0f%%)' %accusgd)
##online    
    plt.plot(scaledx, - w3online[0]/w3online[2] - w3online[1]/w3online[2] * scaledx,
    color = 'y', label = 'online(%.0f%%)' %accuonline)

    plt.legend(bbox_to_anchor=(0.5, 0.1), loc='upper center',
	           ncol=3, borderaxespad=0.,handletextpad = 0.)
    plt.title("Different Updating Rules for Perceptron Learning ")
    fig4.savefig("perceptronlearning.eps")
###### 
##part 4
### out-of-sample evaluation
### perceptron learning
###### 
    print('-' * 60)
    train_index = [25,50,75]
    test_set = X[-25:] # last 25 samples
    test_tar = z[-25:]

### plot
    xx = x[-25:]
    colors = ['r' if t == 0 else 'b' for t in test_tar]
    fig5 = plt.figure(5)
    plt.scatter(xx[:,0], xx[:,1], marker = 'o', c = colors)
## show plot
    scaledx = np.linspace(min(xx[:,0]), max(xx[:,0]), xx.shape[0])
    for i in train_index:
        train_set = X[:i]
        train_tar = z[:i]
        (w,_) = gradient_descent(train_set,train_tar,fun_eva = perceptron_learning)
        res = classification(test_set,w)
        accu = (1.0 - sum(res != test_tar)/float(test_set.shape[0])) * 100.0
##batch
        plt.plot(scaledx, - w[0]/w[2] - w[1]/w[2] * scaledx, 
                    label = 'size=%i(%.0f%%)' %(i,accu))

    lgd = plt.legend(bbox_to_anchor=(0.5, 0.1), loc="upper center",
	           ncol=3, borderaxespad=0.,handletextpad = 0.,
               columnspacing=1)
    plt.title("Different Training Set for out-of-sample evaluation")
    fig5.savefig("outsample.eps")
#    fig5.savefig("outsample.eps",bbox_extra_artists=(lgd,), bbox_inches='tight')

##########
# call main function when executing
##########
if __name__ == "__main__":
    main()



