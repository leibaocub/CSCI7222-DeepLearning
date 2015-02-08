 #!/usr/bin/python
import re
import numpy as np
from neuronlearning import *
from postprocess import *

##########
def main():
##########

#### read data    
    (x,y) = readFile("digits_train.txt")
    X = np.insert(x,0,1,axis=1) # add 1 at the fist column to handle bias term

    (testx,testy) = readFile("digits_test.txt")
    testX = np.insert(testx,0,1,axis=1) # add 1 at the fist column to handle bias term

##########
#### part1 logistic neurons, squared error as the cost function
## classify digit 0-9
## label y as -1 or 1
##########
    tolnum = y.size
    uy = np.unique(y)
    uynum = uy.size
    yy = np.empty((tolnum,uynum))
    yy.fill(-1.0)
    for i in xrange(tolnum):
        yy[i,y[i]] = 1.0

## test set prediction
    tolnum = testy.shape[0]
    testyy = np.empty((tolnum,uynum))
    testyy.fill(-1.0)
    for i in xrange(tolnum):
        testyy[i,testy[i]] = 1.0

###part1,(d)    
    (w, errhis,errhistest) = logistic_neurons(X, yy,
                                        alpha = 0.001, theta = 0.1,
                                        maxiter = 900,
                                        cost_fun = square_error,
                                        active_fun = mytanh,
                                        xtest = testX, ytest=testyy)
#####for ploting cost function history
    fig1 = plt.figure(1)
    iters = np.arange(1,len(errhis)+1)
    plt.plot(iters, errhis)
    iters = np.arange(1,len(errhistest)+1)
    plt.plot(iters, errhistest)
    plt.legend(('train','test'))
    plt.xlabel('epoch')
    plt.ylabel('squared error')
    fig1.savefig("part1_errhis_traintest.eps")

#####compute accuracy
    yp = multiclass_indicator(X,w,fun=mytanh)
    acc = compute_accuracy(yp,y)
    print "train set accuracy %f" %acc

    
    yp = multiclass_indicator(testX,w,fun=mytanh)
    acc = compute_accuracy(yp,testy)
    print "test set accuracy %f" %acc
####################
if __name__ == "__main__":
    main()

