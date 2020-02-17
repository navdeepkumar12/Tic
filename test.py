import nn 

import pm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

X = np.random.rand(10000,2)*3 -1.5
Y = np.array([np.where(np.sum(x*x)>1,[0,1],[1,0]) for x in X])


# input = X
# output = Y

l1 = nn.linear(2,9)
l1.grad_zero()
l1.init_param()
r1 = nn.relu()

l2 = nn.linear(9,2)
l2.grad_zero()
l2.init_param()

s = nn.softmax()
loss = nn.cre()

def model(x,y):
    x = np.array(x)
    y = np.array(y)

    x = l1.forward(x)
    x = r1.forward(x)
    x = l2.forward(x)

    x = s.forward(x)
    dx = loss.forward(x,y)
    dx = s.backward(dx)
    
    dx = l2.backward(dx)
    dx = r1.backward(dx)
    dx = l1.backward(dx)
    
    l1.update()
    l2.update()
    t = np.sum(loss.y)
    return x,y,t

ST = []
st = 0
for i in range(10000):
    j = np.random.choice(range(6617))
    x , y = X[j], Y[j]
    x,y,t = model(x,y)
    st = 0.9*st+0.1*t
    ST.append(st)
    print(t,np.sum(np.abs(x-y)),st)    
index_nn = 1
W = [[l1.w,l1.b],[l2.w,l2.b]]
pickle.dump(W,open('data/W'+str(index_nn), 'wb'))
plt.plot(ST)
plt.savefig('data/L'+str(index_nn)+'.png')