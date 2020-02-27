import numpy as np 
import pm
import tools as tl 
from scipy import signal as sg
x = np.random.randint(0,2,(2,3,4,5))
class pad:
    def dim(m,mode):
        if mode == 'full':
            a,b = m-1,-(m-1)  #x_front,x_back,   p[a:b]=x,  b is negative, a is positive
        if mode == 'valid':
            a,b = 0,0
        if mode == 'same':
            a = int((m-1)/2)
            b = -(m-1-a)  
        return a,b

    def pad1d(x,m,mode='full'):
        x = x.copy()      #filter shape m
        x = np.array(x)
        a,b = pad.dim(m,mode)
        p = np.zeros(x.shape[0]+a-b) # b is negative
        if b==0: b=None
        p[a:b] = x
        return p
    def unpad1d(x,m,mode='full'):
        x = x.copy()
        a,b = pad.dim(m,mode)     #filter shape m
        if b==0: b=None
        x = x[a:b]             # b is negative
        return x
    def pad2d(x,f_shape,mode='full'):
        x = x.copy()
        m,n = f_shape       #filter shape 
        k,l = x.shape
        a,b = pad.dim(m,mode)
        c,d = pad.dim(n,mode)
        p = np.zeros((k+a-b, l+c-d)) # b, d is negative
        if b==0: b=None
        if d==0: d=None    
        p[a:b,c:d] = x
        return p    
    def unpad2d(x,f_shape,mode='full'):
        x = x.copy() 
        m,n = f_shape   #filter shape
        a,b = pad.dim(m,mode)
        c,d = pad.dim(n,mode)
        if b==0: b=None
        if d==0: d=None  
        x =x[a:b,c:d]       # b, d is negative
        return x    
        



class layer:
    def __init__(self):
        self.w = []
        self.b = []
        self.x = []
        self.y = []
        self.dw = []
        self.db = []
        self.dx = []
        self.dy = []

class linear(layer):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        
    def grad_zero(self):
        self.db = np.zeros(self.y_dim)
        self.dw  = np.zeros((self.x_dim, self.y_dim))

    def set_param(self, w, b):
        self.w = w.copy()
        self.b = b.copy()

    def init_param(self):
        self.w = np.random.randn(self.x_dim, self.y_dim)
        self.b = np.zeros(self.y_dim)
   
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x@self.w + self.b
        return self.y

    def backward(self, dy):
        self.dy = dy
        self.dx = self.dy@np.transpose(self.w)  
        self.delta = np.outer(self.x, self.dy)
        self.dw = self.dw*(pm.momentum) + (1-pm.momentum)*self.delta
        self.db = self.db*(pm.momentum) + (1-pm.momentum)*self.dy
        return self.dx

    def update(self):
        self.w = self.w - pm.learning_rate*self.dw      
        self.b = self.b - pm.learning_rate*self.db

        

class hadamard():  
    def __init__(self,m,n):
        self.m = m
        self.n = n
    def grad_zero(self):
        self.dw  = np.zeros((self.m,self.n))

    def set_param(self, w):
        self.w = w
    
    def init_param(self):
        self.w = np.random.randn(self.m,self.n)
       
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x*self.w      
        return self.y

    def backward(self, dy):
        self.dy = dy
        self.dx = self.dy*self.w
        self.delta =self.dy*self.x
        self.dw = self.dw*(pm.momentum) + (1-pm.momentum)*self.delta
        return self.dx

    def update(self):
        self.w = self.w - pm.learning_rate*self.dw      
        
class convolve():  
    def __init__(self,m, mode ='same'):
        #self.mode_list = ['full','same','valid']
        self.m = m
        self.mode = mode
        #self.modeb = self.mode_list[2-self.mode_list.index(self.modef)]   #reversing the mode
        
       
    def grad_zero(self):
        self.delta  = np.zeros(self.m)

    def set_param(self, w):
        self.w = w
    
    def init_param(self):
        self.w = np.random.randn(self.m)
       
    def forward(self,x):
        self.x = x.copy()
        self.X = pad.pad1d(self.x, self.m, self.mode)  #padding
        self.y = sg.correlate(self.X, self.w, mode='valid')     
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = sg.convolve(self.dy, self.w, mode='full')  #padded grad
        self.dw = sg.correlate(self.X, self.dy, mode='valid')
        self.dx = pad.unpad1d(self.dX, self.m, mode=self.mode) #unpadded grad
        return self.dx

    def update(self):
        self.delta = self.delta*(pm.momentum) + (1-pm.momentum)*self.dw
        self.w = self.w - pm.learning_rate*self.dw      

class convolve2d():  
    def __init__(self,shape, mode ='full'):
        self.m, self.n = shape
        self.mode = mode
    
    def grad_zero(self):
        self.delta  = np.zeros((self.m, self.n))

    def set_param(self, w):
        self.w = w
    
    def init_param(self):
        self.w = np.random.randn(self.m, self.n)
       
    def forward(self,x):
        self.x = x.copy()
        self.X = pad.pad2d(x,(self.m, self.n), self.mode)  #padded x = X
        self.y = sg.correlate2d(self.X, self.w, 'valid')     
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = sg.convolve2d(self.dy,self.w, mode='full')        #padded dX
        self.dw = sg.correlate2d(self.X, self.dy, mode='valid')
        self.dx = pad.unpad2d(self.dX, (self.m, self.n), self.mode)   #padded dx
        return self.dx
   
    def update(self):
        self.delta = self.delta*(pm.momentum) + (1-pm.momentum)*self.dw
        self.w = self.w - pm.learning_rate*self.delta    



class convolve3d():  
    def __init__(self, mode ='same'):
        self.mode = mode     #(k,l,m,n) k filter, l input dim, (m,n)filter shape
        self.k, self.l, self.m, self.n = [],[],[],[]
    def grad_zero(self):
        self.delta  =  0 #np.zeros((self.k, self.l, self.m, self.n))

    def set_param(self, w):
        self.w = w
        self.k, self.l, self.m, self.n = np.shape(w)
        self.grad_zero()
    
    def init_param(self,shape):
        self.k, self.l, self.m, self.n = shape
        self.w = np.random.randn(self.k, self.l, self.m, self.n)
        
    def forward(self,x):
        self.x = x.copy()
        self.X = np.array([pad.pad2d(x,(self.m, self.n), self.mode) for x in self.x]) # X = padded x ,dim=(l,m',n')
        self.Y = np.array([[sg.correlate2d(x2,w2,mode='valid') for x2,w2 in zip(self.X, w1)] for w1 in self.w]) # dim=(k,l,m',n')
        self.y = np.sum(self.Y, axis=1) # dim=(k,m',n')       
        return self.y

    def backward(self, dy0):
        self.dy = dy0.copy()
        self.dX = np.array([[sg.convolve2d(dy,w1,mode='full') for w1 in w] for dy,w in zip(self.dy,self.w)]) #padded dX, dim(k,l,m',n')
        self.dw = np.array([[sg.correlate2d(X, dy, mode='valid') for X in self.X] for dy in self.dy]) # dim(k,l,m,n)
        self.dX1 = np.sum(self.dX, axis=0)  #padded dx, dim(l,m',n')
        self.dx = np.array([pad.unpad2d(dX1, (self.m, self.n), self.mode) for dX1 in self.dX1])   #dim(l,m,n)
        return self.dx
    def update(self):
        self.delta = self.delta*(pm.momentum) + (1-pm.momentum)*self.dw
        self.w = self.w - pm.learning_rate*self.delta    


    

class relu(layer):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        self.x = x.copy()
        self.y = np.where(self.x<0,0,self.x)
        return self.y

    def backward(self,dy):
        self.dy = dy.copy()
        self.dx = np.where(self.y<0,0,1)*self.dy
        return self.dx
    
        
class mse():
    def __init__(self):
        pass
    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        self.y = np.sum((self.x - self.label)*(self.x - self.label))/2
        self.dx = self.x - self.label 
        t= np.max(np.abs(self.dx)) 
        if t>10:
            print(self.dx, 'mse:-overflow')
            self.dx = self.dx/t
        
        return self.dx   

class softmax():
    def __init__(self):
        self.dx = []

    def forward(self,x):
        self.x = x.copy()
        self.exp = np.exp(self.x)    
        self.esum = np.sum(self.exp)
        self.y = self.exp/self.esum
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.avg = np.sum(self.dy*self.y)
        self.dx =  self.y*(self.dy - self.avg) 
        t= np.max(np.abs(self.dx)) 
        if t>100:
            #print(self.dx, 'overflow')
            self.dx = self.dx/t
         
        return self.dx 


class cre():
    def __init__(self):
        self.dx = []

    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        if np.sum(np.where(self.x <0,-1,0)) <0 : #sanity check
            tl.cprint('nn.py:cre:- Input are negative for cross entropy loss, input={}'.format(self.x))
            return 
        self.label_entropy = -np.log2(self.label)*self.label
        self.x_entropy = -np.log2(self.x)*self.label
        self.y = self.x_entropy - self.label_entropy
        self.y = np.sum(self.y)   #KL divergence
        self.dx = -self.label*(1/self.x)
        t= np.max(np.abs(self.dx)) 
        if t>100:
            #print(self.dx, 'overflow')
            self.dx = self.dx/t
        
        return self.dx   

class sigmoid():
    def __init__(self):
        self.dx =[]

    def forward(self, x):
        self.x = x.copy()
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = self.dy*self.y*(1-self.y)    
        return self.dx

        
class optimizer():
    def __init__(self, eta=0.001,beta1 = 0.9, beta2 = 0.999,epsilon=10**(-8)):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon =epsilon
        self.t = 0     #time stamp
        self.m = 0    #first order moment aka momentum
        self.v = 0    #second order moment
        self.dw = []
        self.delta = []
    def adam(self, dw):
        self.t = self.t + 1
        self.dw = dw
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.v = self.beta2*self.v + (1-self.beta2)*(self.dw*self.dw)
        self.temp1 = self.m/(1-self.beta1**self.t) #Bias correction, as it was initilized to 0
        self.temp2 = self.v/(1-self.beta2**self.t)  # Bias correction
        self.delta = self.eta*self.temp1/(np.sqrt(self.v)+self.epsilon)
        return self.delta
        
# class sequential():
#     def __init__(self):
#         self.L = []

#     def add_layer(self,name,dim=0):
#         self.L.append([name,dim])        
