import pickle
import numpy as np
import nn
import tools as tl 
import os
import sys
import pm
import matplotlib.pyplot as plt

# index
index = tl.index('data/mnist/index.npy')
cwd = str(os.getcwd())
os.system('cat ' + str(sys.argv[0]) + ' >> data/mnist/'+ str(sys.argv[0]) + str(index) +'.py')
print('mnist:- index = {}'.format(index))


#loading the MNIST data
test = pickle.load(open('data/mnist/test', 'rb'))
train = pickle.load(open('data/mnist/train', 'rb'))
print('MNIST data loaded')

#reshaping
tni = []
tti = []
ttl = []
tnl = []
for a,b in zip(train[0],test[0]):
    a = np.reshape(a,[1,28,28])
    b = np.reshape(b,[1,28,28])
    tni.append(a)
    tti.append(b)
tni = np.array(tni)/256      #normaling input
tti = np.array(tti)/256
# one hot vector
tnl = np.array([np.eye(s+1,10)[s] for s in train[1]])
ttl = np.array([np.eye(s+1,10)[s] for s in test[1]])
print('Data preprocessed')

#  Architecture
conv1 = nn.convolve3d(shape=(5,1,9,9), mode='valid')
add1  = nn.add()
relu1 = nn.relu()
conv2 = nn.convolve3d(shape=(10,5,20,20), mode='valid')
add2 = nn.add()
lin = nn.linear()
softmax = nn.softmax()
cre = nn.cre()
print('Architecture loaded')

# #param init randoom
if pm.initial_Q == False:
    conv1.set_param(np.random.randn(5,1,9,9)/(np.sqrt(81*2)))
    conv2.set_param(np.random.randn(10,5,20,20)/(np.sqrt(5*19*19*2)))
    lin.init_param((10,10)) 
    print('mnist:- Weights initialized randomly')
else:
    if  pm.initial_Q == True:
        ind = index -1
        W = tl.load_Q('data/mnist/param'+str(ind))
    else:
        W = tl.load_Q(pm.initial_Q)   
    # set param from initial values
    #W = [conv1.w, add1.w, conv2.w, add2.w, lin.w, lin.b]
    c1, a1, c2, a2, lw, lb = W
    conv1.set_param(c1)
    conv2.set_param(c2)
    lin.set_param(lw,lb)
    add1.set_param(a1)
    add2.set_param(a2)


Accuracy = []
#compute  graph
def model(data, update = True):
    img, label = data
    acc = []
    for x,y in zip(img, label):

        x = conv1.forward(x)
        x = add1.forward(x)
        x = conv2.forward(x)
        x = add2.forward(x)
        x = x.reshape([10])
        x = lin.forward(x)
        x = softmax.forward(x)
        loss = cre.forward(x,y)
        
        if update ==True:
            dx = cre.backward()
            dx = softmax.backward(dx)
            dx = lin.backward(dx)
            dx = dx.reshape([10,1,1])
            dx = add2.backward(dx)
            dx = conv2.backward(dx)
            dx = add1.backward(dx)
            dx = conv1.backward(dx)
            
            conv1.update()
            conv2.update()
            add1.update()
            add2.update()
            lin.update()
        
        
        pred =  np.argmax(x)
        l = np.argmax(y)
        if pred ==l:
            acc.append(1)
        else:
            acc.append(0)
        ac = np.mean(acc)
        Accuracy.append(ac)

        tl.fprint('pred={}, label = {},   acc = {:.4f},   loss = {:.4f}'.format(pred,l,ac, loss))  
        if np.amax(np.abs(conv1.w)) > 5:
            tl.cprint('overflow')
            break
    return acc     

for i in range(5):
    acc = model((tni,tnl))
    tl.cprint('\n TicTac:mnist:- Training accuracy = {}, epoch ={}\n'.format(np.mean(acc),i), color='blue')
    W = [conv1.w, add1.w, conv2.w, add2.w, lin.w, lin.b]
    Acc = model((tti,ttl), update=False) 
    tl.cprint('\n TicTac:mnist:- Test accuracy = {}, epoch = {}\n'.format(np.mean(Acc),i), color='blue')
pickle.dump(W,open('data/mnist/param'+str(index), 'wb')) 
plt.plot(Accuracy)
plt.savefig('data/mnist/acc'+str(index)+'.png')
print('Trained weights saved at data/mnist/param{}'.format(index))    