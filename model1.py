import pickle
import numpy as np
import nn
import tools as tl 
import os
import sys
import pm
import matplotlib.pyplot as plt

#  Architecture
conv1 = nn.convolve3d(shape=(5,1,3,3), mode='valid')
add1  = nn.add()
relu1 = nn.relu()
conv2 = nn.convolve3d(shape=(1,5,3,3), mode='valid')
add2 = nn.add()
lin = nn.linear()
sigmoid = nn.sigmoid()
mse = nn.mse()
print('Architecture loaded')

layer = [conv1, add1, relu1, conv2, add2, lin, sigmoid, mse]

#compute  graph
def model(x,y, update = True):
    x = conv1.forward(x)
    x = add1.forward(x)
    x = conv2.forward(x)
    x = add2.forward(x)
    x = x.reshape([256])
    x = lin.forward(x)
    x = sigmoid.forward(x)
    x = mse.forward(x,y)
    
    if update ==True:
        dx = mse.backward()
        dx = sigmoid.backward(dx)
        dx = lin.backward(dx)
        dx = dx.reshape([1,16,16])
        dx = add2.backward(dx)
        dx = conv2.backward(dx)
        dx = add1.backward(dx)
        dx = conv1.backward(dx)
        
        conv1.update()
        conv2.update()
        add1.update()
        add2.update()
        lin.update()
    
    return 
    
    
