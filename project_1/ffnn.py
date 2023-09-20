import numpy as np
import gzip
import pickle
from main import *
import random
import time
# Activation function, parameters: double
def sigmoid_func(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_deriv_func(x):
    return sigmoid_func(x)*(1.0 - sigmoid_func(x))


class ffnn:
    def __init__(self, inputLayer_width, hiddenLayer_width, outputLayer_width):
        self.bias_inputlayer = np.zeros((inputLayer_width, inputLayer_width))
        self.weight_inputlayer = np.random.randn(
            inputLayer_width, hiddenLayer_width)*0.1
        self.weight_hiddenlayer = np.random.randn(
            hiddenLayer_width, outputLayer_width)*0.1
        self.bias_outputlayer = np.zeros((outputLayer_width, outputLayer_width))

        self.model = [self.bias_inputlayer, self.weight_inputlayer,
                      self.weight_hiddenlayer, self.bias_outputlayer]
    
    """
    Backwards propagation with a fixed layer width of 3.
    Propagates through the network, finding the partial derivatives of the weights and biases. And then returns the gradient. Which is then used to tune the weights and biases.


    """
    def backward(self, X, Y):
        T = 3
        a,o=[[]]*T,[[]]*T
        o[0]=X
        
        for t in range(1, T):       
            a[t]=np.dot(self.model[t].T,np.array(o[t-1]))
            o[t]=sigmoid_func(a[t])
            if t==1:
                a[1][len(self.model[t].T)-1]=1
                o[1][len(self.model[t].T)-1]=sigmoid_func(1)
  
        delta=[[]]*T
        delta[T-1]=(o[T-1]-Y)

        fpri=np.vectorize(sigmoid_deriv_func)
        for t in reversed(range(1, T-1)):
            aa=fpri(a[t+1]) 
            delta[t]=np.dot(self.model[t+1],np.multiply(aa,delta[t+1]))
    
        gradient=[[]]*4
       
        for t in range(1,T):
            aa=fpri(a[t])
            A_delta=np.reshape(np.multiply(aa,delta[t]),(-1,1))
            oo=np.reshape(o[t-1],(1,-1))
            gradient[t]=np.matmul(A_delta,oo).T
        return gradient
    """
    Forward function, carries the input X through the network, and returns the output.
    """
    def forward(self, X):
        X = np.append(X, 1.0)
        T = 3
        a=[[]]*T
        o=[[]]*T
        o[0]=X
        for t in range(1, T):       
            a[t]=np.dot(self.model[t].T,np.array(o[t-1]))
            o[t]=sigmoid_func(a[t])
            if t==1:
                a[1][len(self.model[t].T)-1]=1
                o[1][len(self.model[t].T)-1]=sigmoid_func(1)
        return np.exp(a[T-1])/np.sum(np.exp(a[T-1]))
    """
    Square loss function, returns the summed meansquare of the error, and an accuracy. where x is the estimated value, and y is the expected value. They are both of the type np.ndarray.
    """
    def validate(self, x, y):
        result= accresult = 0
        for index in range(len(x)):
            predicted_val = self.forward(x[index]).argmax()
          #  predicted_val = np.array(predicted).argmax()
            if predicted_val == y[index]:
                accresult += 1
            result += (np.divide(
                np.power((y[index] - predicted_val), 2), 2*len(x)))
        return result, accresult/len(y)
      
      
        """
        The training loop, with adjustable Epochs, batchsize, and learnrate.
        The parameters x_train, y_train, x_valid, y_valid are all of the type numpy.ndarray

        """
    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch, learnrate):
        gradient=self.model*0
        for epoch_nbr in range(epochs):
            starttime = time.time()
            randomize=np.arange(len(y_train))
            x=x_train[randomize]
            y=y_train[randomize]
            for k in range(0,len(y),batch):
                y_batch, x_batch=y[k:k+batch], x[k:k+batch]
                length_current_batch=len(y_batch)
                for i in range(len(y_batch)):
                    x_biased=np.append(x_batch[i], 1.0)
                    gradient=self.backward(x_biased, y_batch[i])
                self.model[1] -=  learnrate * gradient[1]/length_current_batch
                self.model[2] -=  learnrate * gradient[2]/length_current_batch        
            val, acc = self.validate(x_valid, y_valid)
            print("Epoch ", epoch_nbr + 1, "/", epochs,
            "with validation loss:", val,"Accuracy: ",round(acc*100, 2),"%", "Time for Epoch:", time.time()-starttime)
