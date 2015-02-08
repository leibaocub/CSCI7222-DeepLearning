 #!/usr/bin/python
import numpy as np
from neuralnetwork import *
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
#    X = np.insert(x,0,1,axis=1) # add 1 at the fist column to handle bias term

    (testx,testy) = readFile("digits_test.txt")
#    testX = np.insert(testx,0,1,axis=1) # add 1 at the fist column to handle bias term


    output_fun = mysigmoid ## mysigmoid, mysoftmax
    hidden_fun = mysigmoid # mytanh
    cost_fun = square_error
    (w, errhis) = neural_network(x,y,alpha = 0.003, theta = 0.0, 
                                maxiter = 1000, cost_fun = cost_fun,
                                hidden_fun = hidden_fun,
                                hidden_units = [20, 15],
                                output_fun = output_fun)
    print errhis[-1]

#####for ploting cost function history
    fig1 = plt.figure(1)
    iters = np.arange(1,len(errhis)+1)
    plt.plot(iters, errhis)
    plt.legend(['train'])
    plt.xlabel('epoch')
    plt.ylabel(cost_fun.__name__)
    fig1.savefig("part3_errhis_traintest.eps")

####training set
    yp = nn_predict(w, x, output_fun)
    acc = compute_accuracy(yp,y)
    print "train set accuracy %f" %acc

####test set
    yp = nn_predict(w, testx, output_fun = output_fun, hidden_fun = hidden_fun)
    acc = compute_accuracy(yp, testy)
    print "test set accuracy %f" %acc

####################
if __name__ == "__main__":
    main()
                



