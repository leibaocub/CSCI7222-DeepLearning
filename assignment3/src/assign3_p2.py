 #!/usr/bin/python
import numpy as np
from neuronlearning import *
from postprocess import *

##################################################
# Assignment 3
###part3
# handprinted digits recognition
##################################################
##########
def main():
#### read data    
    (x,y) = readFile("digits_train.txt")
    X = np.insert(x,0,1,axis=1) # add 1 at the fist column to handle bias term

    (testx,testy) = readFile("digits_test.txt")
    testX = np.insert(testx,0,1,axis=1) # add 1 at the fist column to handle bias term

##label y as 0 or 1
    tolnum = y.size
    uy = np.unique(y)
    uynum = uy.size
    yy = np.zeros((tolnum,uynum))
    for i in xrange(tolnum):
        yy[i,y[i]] = 1.0

## test set prediction
##label testy as 0 or 1
    tolnum = testy.shape[0]
    testyy = np.zeros((tolnum,uynum))
    for i in xrange(tolnum):
        testyy[i,testy[i]] = 1.0

    (w, errhis,errhistest) = logistic_neurons(X, yy, xtest = testX, ytest = testyy,
                                        alpha = 0.02, theta = 0.5, 
                                        maxiter = 100)

#####for ploting cost function history
    fig1 = plt.figure(1)
    iters = np.arange(1,len(errhis)+1)
    plt.plot(iters, errhis)
    iters = np.arange(1,len(errhistest)+1)
    plt.plot(iters, errhistest)
    plt.legend(('train','test'))
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')
    fig1.savefig("part2_errhis_traintest.eps")

####training set
    yp = multiclass_indicator(X,w,fun=mytanh)
    acc = compute_accuracy(yp,y)
    print "train set accuracy %f" %acc

####test set
    yp = multiclass_indicator(testX,w,fun=mytanh)
    acc = compute_accuracy(yp,testy)
    print "test set accuracy %f" %acc

####################
if __name__ == "__main__":
    main()
                



