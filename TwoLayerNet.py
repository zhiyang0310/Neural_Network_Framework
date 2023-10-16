import numpy as np
from ComputeLayers import *
import collections
import Functions

class TwoLayerNet:
    def __init__(self,input_size,hidden_size1,output_size,weight_ini_std = 0.01):
        # parameter
        self.params = {}
        self.params['W1'] = weight_ini_std*np.random.randn(32,1,5,5)
        self.params['b1'] = np.zeros(32)
        self.params['W2'] = weight_ini_std * np.random.randn(64, 32, 7, 7)
        self.params['b2'] = np.zeros(64)
        # self.params['W2'] = weight_ini_std*np.random.randn(input_size,hidden_size1)
        # self.params['b2'] = np.zeros((1,hidden_size1))
        # self.params['W3'] = weight_ini_std * np.random.randn(hidden_size1, output_size)
        # self.params['b3'] = np.zeros((1,output_size))

        #compute layers
        self.set_layers()

    def set_layers(self):
        self.layers = collections.OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],1,0)
        self.layers['ReLu1'] = ReLu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], 1, 0)
        self.layers['ReLu2'] = ReLu()
        self.layers['Capsule'] = CapsuleForCorrelation(cap_dim = 8)
        self.layers['Unit'] = UnitForCorrelation()
        self.layers['Correlation'] = Correlation(8)


        # self.layers['Pooling1'] = Pooling(2,2,2,0)
        # self.layers['Fullconnection1'] = Fullconnection()
        # self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        # self.layers['ReLu2'] = ReLu()
        # self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # self.lastlayer = Softmaxwithloss()
        self.lastlayer = LosswithNorm2()

    def set_params(self,params):
        for param in params.keys():
            self.params[param] = params[param]
        self.set_layers()

    def update_params(self,delta):
        for param in delta.keys():
            self.params[param] = self.params[param] + delta[param]
        self.set_layers()

    def predict(self,x):
        for layer in self.layers.values():
            if isinstance(layer,Dropout):
                x = layer.forward(x,False)
            else:
                x = layer.forward(x)
        return x

    def loss(self,x,label):
        x = self.predict(x)
        # attention
        return self.lastlayer.forward(x)
        # return self.lastlayer.forward(x,label)

    def gradient(self,x,label):
        # forward
        self.loss(x,label)
        # backward
        dx = self.lastlayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dx = layer.backward(dx)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dw
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dw
        grads['b2'] = self.layers['Conv2'].db
        # grads['W2'] = self.layers['Affine1'].dw
        # grads['b2'] = self.layers['Affine1'].db
        # grads['W3'] = self.layers['Affine2'].dw
        # grads['b3'] = self.layers['Affine2'].db
        return grads

    def numerical_gradient(self,x,t):
        params = list(self.params.keys())
        grads = {}
        h = np.e ** -4
        for param in params:
            dparam = np.zeros_like(self.params[param])
            for i in range(self.params[param].shape[0]):
                for j in range(self.params[param].shape[1]):
                    self.params[param][i,j] += h
                    a = self.loss(x,t)
                    self.params[param][i,j] -= 2*h
                    b = self.loss(x,t)
                    self.params[param][i,j] += h
                    dparam[i,j] = (a - b)/(2*h)
            grads[param] = dparam
        return grads

    def accuracy(self,x,labels):
        count = 0
        y = self.predict(x)
        for i in range(y.shape[0]):
            if np.argmax(y[i,:]) == np.argmax(labels[i,:]):
                count += 1
        return count/y.shape[0]