import pickle
import numpy as np
import nn
import tools as tl 
import os
import sys
import pm
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath('home/navdeep/RLC/Robot-Learning-and-Control'))


x = np.random.randint(0,3,(1,5,5))
y = np.where(x>0,0,1)
y = y.reshape([25])
#  Architecture
conv1 = nn.convolve3d(shape=(10,1,3,3), mode='same')
add1  = nn.add()
relu1 = nn.relu()
conv2 = nn.convolve3d(shape=(1,10,3,3), mode='same')
add2 = nn.add()
lin = nn.linear((25,25 ))
sigmoid = nn.sigmoid()
mse = nn.mse()
print('Architecture loaded')

# weigths init


layer = [conv1, add1, relu1, conv2, add2, lin, sigmoid, mse]

#compute  graph
def model(x,y, update = True):
    x = conv1.forward(x)
    x = add1.forward(x)
    x = conv2.forward(x)
    x = add2.forward(x)
    #x = x.reshape([25])
    x = lin.forward(x)
    x = sigmoid.forward(x)
    x = mse.forward(x,y)
    
    if update ==True:
        dx = mse.backward()
        dx = sigmoid.backward(dx)
        dx = lin.backward(dx)
        #dx = dx.reshape([1,5,5])
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
    
    
