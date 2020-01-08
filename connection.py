import math 
import numpy as np 


class Connecetion:
    def __init__(self,connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weigth = np.random.normal()
        self.dWeight = 0.0

class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self,layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connecetion(neuron)
                self.dendrons.append(con)
    
    def addError(self,err):
        self.error += err
    
    def sigmoid(self,x):
        o = 1/(1+math.exp(-x))
        return o 

    def dSigmoid(self,x):
        o = x * (1.0 -x)
        return o
    
    def setError(self,err):
        self.error = err
    
    def setOutput(self,output):
        self.output=output
    
    def getOutput(self):
        return self.output

    def feedForword(self):
        sumout = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumout += dendron.connectedNeuron.getOutput() *dendron.weigth
        self.output = self.sigmoid(sumout)