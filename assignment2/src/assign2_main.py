 #!/usr/bin/python
import re
import numpy as np
from neuralnetworkAux import *

##################################################
# Assignment 2
# handprinted digits recognition
##################################################
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

##########
def main():
#### read data    
    (x,y) = readFile("digits_train.txt")
    X = np.insert(x,0,1,axis=1) # add 1 at the fist column to handle bias term

    (testx,testy) = readFile("digits_test.txt")
    testX = np.insert(testx,0,1,axis=1) # add 1 at the fist column to handle bias term

##########
#### part2 recognition
## label 2 or not 2
##########
    tolnum = y.size
## get label: class 2 -> label 1
##            not class 2 -> label 0
    ind2 = np.nonzero(y == 2)[0]
    mask2 = np.zeros((tolnum,1))
    mask2[ind2] = 1.0

    (w,errhis) = gradient_descent(X, mask2, alpha = 1.0, 
                            fun_eva = perceptron_learning,
                            batchsize = 1,
                            maxiter = 200)
    plt.show()
    res = classification(X,w)
    acc = compute_accuracy(res, mask2)
    print "train set accuracy %f" %acc

## get label: class 2 -> label 1
##            not class 2 -> label 0
    testind2 = np.nonzero(testy == 2)[0]
    mask2 = np.zeros((testy.shape[0],1))
    mask2[testind2] = 1.0
    res = classification(testX,w)
    acc = compute_accuracy(res, mask2)
    print "test set accuracy %f" %acc
    plot_confusionmatrix(mask2,res,['2','not 2'],'part2.eps')

##########
#### part 3
## classify lable 8 and 0
##########
    ind0 = np.nonzero( y == 0)[0]
    ind8 = np.nonzero( y == 8)[0]
    ind08 = np.concatenate((ind0,ind8))

    tol0 = ind0.shape[0]
    tol8 = ind8.shape[0]
    tol08 = tol0 + tol8
## get label: class 0 -> label 1
##            class 8 -> label 0

    mask08 = np.zeros((tol08,1))
    mask08[:tol0] = 1.0

    X08 = X[ind08]
    (w08,_) = gradient_descent(X08, mask08, alpha = 1.0, 
                            fun_eva = perceptron_learning,
                            batchsize = 1,
                            maxiter = 20)
    res = classification(X08,w08)
    acc = compute_accuracy(res,mask08)
    print "train set accuracy %f" %acc

### test set prediction
    ind0 = np.nonzero( testy == 0)[0]
    ind8 = np.nonzero( testy == 8)[0]
    ind08 = np.concatenate((ind0,ind8))

    tol0 = ind0.shape[0]
    tol8 = ind8.shape[0]
    tol08 = tol0 + tol8

## get label: class 0 -> label 1
##            class 8 -> label 0

    mask08 = np.zeros((tol08,1))
    mask08[:tol0] = 1.0

    testX08 = testX[ind08]
    res = classification(testX08,w08)
    acc = compute_accuracy(res,mask08)
    print "test set accuracy %f" %acc
    plot_confusionmatrix(mask08,res,['0','8'],'part3.eps')


##########
#### part 4
## classify all samples
##########
    uy = np.unique(y)
    uynum = uy.size
    yy = np.zeros((tolnum,uynum))
    for i in xrange(tolnum):
        yy[i,y[i]] = 1.0

    (wall,_) = gradient_descent(X, yy, alpha = 1.0, 
                            fun_eva = multiclass_perceptron,
                            batchsize = 1,
                            maxiter = 500)
    res = multiclass_classification(X,wall)
    acc = compute_accuracy(res,yy)
    print "train set accuracy %f" %acc


## test set prediction
    tolnum = testy.shape[0]
    yy = np.zeros((tolnum,uynum))
    for i in xrange(tolnum):
        yy[i,testy[i]] = 1.0

    res = multiclass_classification(testX,wall)
    acc = compute_accuracy(res,yy)
    print "test set accuracy %f" %acc

    res2label = np.nonzero(res)[1]
    labels = ["%d" %i for i in xrange(uynum)]
    plot_confusionmatrix(testy,res2label, labels ,'part4.eps')

if __name__ == "__main__":
    main()
                



