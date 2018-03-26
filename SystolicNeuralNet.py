import numpy as np
from sklearn import utils
import time
from libs import *

np.random.seed(7)

def np_map(func, x):
    for i in range(0, len(x)):
        x[i] = func(x[i])
    return x

activations = {
    'sigmoid':
                {
                    'n': lambda x: 1/(1 + np.exp(-1*x)),
                    'd': lambda x: x * ( 1 - x),
                }
    }

forward_times = []
backward_times = []
class Dense:
    def __init__(self, hidden_nodes, activation = 'sigmoid', bias = 0):
        self.hidden_size = hidden_nodes
        self.activation = activations[activation]['n']
        self.activation_deriv = activations[activation]['d']
        self.bias = bias
    def __str__(self):
        return 'Dense(num_hidden_nodes = %s)' %(self.hidden_size)

class NeuralNet:
    def __init__(self, input_size = (), lr = 0.5):
        self.prev_size = input_size[0]
        self.input_size = input_size[0]
        self.layers = {}
        self.lr = lr
        self.i = 0
        self.sys_size = 0

    def add(self, Model):
        self.i+=1
        self.layers[self.i] ={
            'Model': Model,
            'weights': 2*np.random.randn(Model.hidden_size, self.prev_size) - 1,
            'new_weights': 0,
            'output': 0,
            'error': [],
        }
        self.prev_size = Model.hidden_size

    def train(self, x_train,y_train, epochs = 1):
        error = []
        self.sys_size = find_optimum_shape_Neural_Net(net.layers, verbose = True)
        for i in range(epochs):
            error_stuff = 0
            it = 0
            for x,y in zip(x_train,y_train):
                x = np.array(x).reshape((-1,1))
                y = np.array(y).reshape((-1,1))
                prediction = self.forward_prop(x)
                self.back_error(x, self.i, prediction, y)
                for k in self.layers.keys():
                    self.layers[k]['weights'] = self.layers[k]['new_weights']
                    self.layers[k]['new_weights'] = 0
                error_stuff += (0.5*(y - prediction)**2)[0][0]
                it +=1
            print('Epoch %s: %s' %(i, error_stuff/it))
            error.append( error_stuff/it )

        #import matplotlib.pyplot as plt
        #plt.plot(error)
        #lt.show()

    def back_error(self, training_data, I, output = 0, real = 0):
        global backward_times

        if I > 0:
            input1 = training_data
            if(I != 1):
                input1 = self.layers[I-1]['output']
            if(I == self.i):
                error = -(real-output)
                new = np.multiply(error, self.layers[I]['Model'].activation_deriv(output))
                self.layers[I]['error'] = new
            else:
                start = time.time()
                previous_layer_error = np.dot(self.layers[I+1]['weights'].T, self.layers[I+1]['error'])
                end = time.time()
                backward_times.append(['Layer%s'%(I), end - start])
                error = np.multiply(previous_layer_error, self.layers[I]['Model'].activation_deriv(self.layers[I]['output']) )
                self.layers[I]['error'] = error
            self.layers[I]['new_weights'] = self.layers[I]['weights'] - self.lr * np.dot(self.layers[I]['error'], input1.T)
            self.back_error(training_data=training_data, I = I - 1)

    def forward_prop(self,result, I = 1, ):
        global forward_times

        if(I == self.i + 1):
            return result
        else:
            #dot = np.dot(self.layers[I]['weights'], result)
            dot, trackiterations, clocks, for_freq, total_time = blockNormalSystolicMultiply(initNormalSystolic(self.sys_size, 1), self.layers[I]['weights'], result, verbose = False)
            output = self.layers[I]['Model'].activation(dot +self.layers[I]['Model'].bias)
            forward_times.append(['Layer%s'%(I), total_time])
            print('Finished Multiplying Layer%s weights'%(I))
            self.layers[I]['output'] = output
            return self.forward_prop( result = output, I = I + 1, )

    def __str__(self):
        string = 'Input_dims: %s ' % ( self.input_size)
        string += '\n' + ''.join(['-' for i in range(0, 20)])
        for k,v in self.layers.items():
            string += '\n' + str(k) + ':' + str(v['Model']) + ', ' + 'weight_size: ' + str(v['weights'].shape)
        return string

def data_reader(filename):
    f = open(filename, mode ='r')
    s = f.readline()
    x_train = []
    y_train = []
    while s != '':
        vals = s.split(',')
        x_train.append(list( map (lambda t: float(t), filter(lambda x:  x != '', vals[0:-1]) ) ) )
        y_train.append(float(vals[-1].rstrip()) )
        s = f.readline()
    return np.array(x_train), np.array(y_train)

def shuffle(array):
    for i in range(len(array) - 1):
        rand = np.random.randint(low = i, high=len(array))
        array[i], array[rand] = array[rand], array[i]
    return array

#import os
#os.chdir('C:\\Users\\Sachin Konan\\Documents\\PythonPractice')
#x_train, y_train = data_reader('seeds.csv')

#X,Y = utils.shuffle(x_train, y_train)