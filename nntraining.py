import nn 
import tools as tl 
import pm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

S = tl.load_Q('state')
Q = tl.load_Q('data/Q0')
X = [np.array(list(map(int,s))) for s in S] 
V = [10*Q[s] for s in S]
Y = [nn.softmax().forward(v) for v in V]

# Z = [np.where(x >0,0,1) for x in X ]
# Y = [z*y for z,y in zip(Z,Y)]
# Y = [y/np.sum(y) for y in Y]
if os.path.exists('index_nn.npy'):
    index_nn = np.loadtxt('index_nn.npy')
    index_nn = int(index_nn) +1
else :
    index_nn = 10
np.savetxt('index_nn.npy', [index_nn])
    
#W = tl.load_Q('data/W'+str(index_nn-1))
# input = X
# output = Y

l1 = nn.linear(9,9)
l1.grad_zero()
l1.init_param()
r1 = nn.relu()

l2 = nn.linear(9,9)
l2.grad_zero()
l2.init_param()
r2 = nn.relu()

l3 = nn.linear(9,9)
l3.grad_zero()
l3.init_param()

s = nn.softmax()
loss = nn.mse()

def model(x,y):
    x = np.array(x)
    y = np.array(y)

    x = l1.forward(x)
    x = r1.forward(x)
    x = l2.forward(x)
    x = r2.forward(x)
    
    x = l3.forward(x)

    x = s.forward(x)
    dx = loss.forward(x,y)
    dx = s.backward(dx)
    
    dx = l3.backward(dx)

    dx = r2.backward(dx)
    dx = l2.backward(dx)
    dx = r1.backward(dx)
    dx = l1.backward(dx)
    
    l1.update()
    l2.update()
    l3.update()
    t = np.sum(loss.y)
    return x,y,t
TT =[]
SS= []
for i in range(1000):
    T = []
    S = []

    for x,y in zip(X,Y):
        x1,y1,t = model(x,y)
        T.append(t)
        if np.argmax(x1) == np.argmax(y1):
            S.append(1)
        else:
            S.append(0) 
    TT.append(np.mean(T))
    SS.append(np.mean(S))
    tl.cprint('\n delta ={}'.format(np.mean(np.abs([l1.delta,l2.delta,l3.delta]))))
    tl.cprint('loss ={}, acc = {}\n'.format(np.mean(T), np.mean(S)))           
    # st = 0.9*st+0.1*t
    # ST.append(st)
    # print(t,np.sum(np.abs(x-y)),st)    

W = [[l1.w,l1.b],[l2.w,l2.b],[l3.w,l3.b]]
pickle.dump(W,open('data/W'+str(index_nn), 'wb'))
plt.plot(TT)
plt.savefig('data/L'+str(index_nn)+'.png')
plt.plot(SS)
plt.savefig('data/acc'+str(index_nn)+'.png')