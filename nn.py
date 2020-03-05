import numpy as np 
import pm
import tools as tl 
from scipy import signal as sg

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
        
class optimizer():
    def __init__(self, eta=0.001,beta1 = 0.9, beta2 = 0.999,epsilon=10**(-8)):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon =epsilon
        self.t = 0     #time stamp
        self.m = 0    #first order moment aka momentum
        self.v = 0    #second order moment
        self.dw = None
        self.delta = []
        self.beta1t = 1
        self.beta2t = 1
        

class momentum(optimizer):
    def __init__(self):
        super().__init__()
    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.delta = self.eta*self.temp3
        return self.delta


class adam(optimizer):
    
    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw.copy()
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.v = self.beta2*self.v + (1-self.beta2)*(self.dw*self.dw)
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t
        self.beta2t = self.beta2t*self.beta2      # = self.beta2**t
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.temp4 = self.v/(1-self.beta2t)  # =v,  Bias correction
        self.delta = self.eta*self.temp3/(np.sqrt(self.temp4)+self.epsilon)
        return self.delta

class adamax(optimizer):
    def __init__ (self):
        super().__init__()
        self.first_time = True
        
    def initilize(self):
        if self.first_time == True:
            self.shape = self.dw.shape
            self.v = np.zeros(self.shape) 
            self.first_time == False   

    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw ; self.initilize()    # takes shape of dw and makes v same shape
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.v = np.max(np.array([self.beta2*self.v, np.abs(self.dw)]), axis=0) # = max(beta*v,|dw|) 
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t, for m correction
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.temp4 = self.v        # =v , Bias correction not required, as it gets first input from real data
        self.delta = self.eta*self.temp3/(self.temp4+self.epsilon)  
        return self.delta


class param():
    def __init__(self):
        self.shape = None
        self.w =  []

class ones(param):
    def forward(self,shape):
        self.shape = shape    
        self.w = np.ones(self.shape)
        return self.w

class zeros(param):
    def forward(self,shape):
        self.shape = shape
        self.w = np.zeros(self.shape)  
        return self.w

class uniform(param):
    def forward(self,shape):
        self.shape = shape
        self.w = np.random.random(self.shape)
        return self.w

class normal(param):
    def forward(self,shape):
        self.shape = shape
        self.w = eval('np.random.randn'+ str(self.shape))
        return self.w

class he(param):
    def forward(self,shape):
        self.shape = shape 
        self.scale = np.sqrt(self.shape[0],np.prod(self.shape[1:]))   # for lin and conv2d, conv3d, ?for conv1d, add1
        self.w = eval('np.random.randn'+ str(self.shape))/self.scale
        print('nn:param:he:- param initilized HE random normal')
        return self.w


class layer():

    def __init__(self, shape=None, opt='adam', param = 'normal', trainable = True, mode = 'valid'):
        self.x, self.y, self.dy, self.dx , self.dw, self.delta = np.repeat(None,6)
        self.w = 0 # By defalut weights is 0 if shape is not specified, works good for add layer
        self.opt = eval(opt+'()')
        self.shape = shape
        self.trainable = trainable
        self.mode = mode #only for convolution
        # Set param and sanity check
        if type(param) ==str:  #  param init is inputed
            self.param = eval(param+'()')
            if shape != None:  # Dosn't init weights of Relu, cre , etc
                self.init_param()
        if type(param) in {list, np.ndarray}: # set param and sanity check
            if self.shape == None or self.shape == param.shape:
                self.set_param(param)         
            else: tl.cprint('layer:init:- layer shape ={} not matched with param shape {}'.format(self.shape,param.shape))
    
    def set_opt(self,opt):
        self.opt = opt
    
    def set_param(self, w):
        self.w = w
        self.shape = w.shape
        
    def init_param(self, shape = None, param=None): 
        if param != None:   # changing __init__ param init
            self.param = eval(param+'()')   
        if shape != None: # changing __init__ param shape
            self.shape = shape    
        self.w = self.param.forward(self.shape)
    
    def forward(self,x):
        pass
   
    def backward(self,dy):
        pass
   
    def update(self):
        if self.trainable:
            self.delta = self.opt.forward(self.dw)
            self.w = self.w - self.delta


class linear(layer):
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x@self.w
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = self.dy@np.transpose(self.w)  
        self.dw = np.outer(self.x, self.dy)
        return self.dx
    

 
class relu(layer):
    def __init__(self):
        super().__init__(trainable=False)
        
    def forward(self,x):
        self.x = x.copy()
        self.y = np.where(self.x<0,0,self.x)
        return self.y

    def backward(self,dy):
        self.dy = dy.copy()
        self.dx = np.where(self.y<0,0,1)*self.dy
        return self.dx
    
        
class mse(layer):
    def __init__(self):
        super().__init__(trainable=False)
    
    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        self.y = np.sum((self.x - self.label)*(self.x - self.label))/2
        return self.y

    def backward(self):
        self.dx = self.x - self.label         
        return self.dx   

class softmax(layer):
    def __init__(self):
        super().__init__(trainable=False)
    
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
        return self.dx 


class cre(layer):
    def __init__(self):
        super().__init__(trainable=False)
        
    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        self.label_entropy = -np.log2(self.label+0.0001)*self.label    #0.0001 is added to avoid log(0)
        self.x_entropy = -np.log2(self.x+0.0001)*self.label
        self.y = self.x_entropy  - self.label_entropy
        self.y = np.sum(self.y)   #KL divergence
        return self.y
    
    def backward(self) :   
        self.dx = -self.label*(1/(self.x+0.0001))       # 0.0001 is added to avoid insanely large value of 1/x     
        return self.dx   

class sigmoid(layer):
    def __init__(self):
        super().__init__(trainable=False)
    
    def forward(self, x):
        self.x = x.copy()
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = self.dy*self.y*(1-self.y)    
        return self.dx

class add(layer):
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x + self.w
        return self.y
    
    def backward(self, dy): 
        self.dy = dy.copy()
        self.dw = self.dy 
        self.dx = self.dy
        return self.dx  

class hadamard(layer):    
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x*self.w      
        return self.y

    def backward(self, dy):
        self.dy = dy
        self.dx = self.dy*self.w
        self.dw =self.dy*self.x
        return self.dx

        
class convolve(layer):  
    def forward(self,x):
        self.x = x.copy()
        self.m = self.shape
        self.X = pad.pad1d(self.x, self.m, self.mode)  #padding
        self.y = sg.correlate(self.X, self.w, mode='valid')     
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = sg.convolve(self.dy, self.w, mode='full')  #padded grad
        self.dw = sg.correlate(self.X, self.dy, mode='valid')
        self.dx = pad.unpad1d(self.dX, self.m, mode=self.mode) #unpadded grad
        return self.dx

    
class convolve2d(layer):  
    def forward(self,x):
        self.x = x.copy()
        self.m, self.n = self.shape
        self.X = pad.pad2d(x,(self.m, self.n), self.mode)  #padded x = X
        self.y = sg.correlate2d(self.X, self.w, 'valid')     
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = sg.convolve2d(self.dy,self.w, mode='full')        #padded dX
        self.dw = sg.correlate2d(self.X, self.dy, mode='valid')
        self.dx = pad.unpad2d(self.dX, (self.m, self.n), self.mode)   #padded dx
        return self.dx
   
    


class convolve3d(layer):          
    def forward(self,x):
        self.x = x.copy()
        #(k,l,m,n) k filter, l input dim, (m,n)filter shape
        self.k, self.l, self.m, self.n = self.shape
        self.X = np.array([pad.pad2d(x,(self.m, self.n), self.mode) for x in self.x]) # X = padded x ,dim=(l,m',n')
        self.Y = np.array([[sg.correlate2d(x2,w2,mode='valid') for x2,w2 in zip(self.X, w1)] for w1 in self.w]) # dim=(k,l,m',n')
        self.y = np.sum(self.Y, axis=1) # dim=(k,m',n')       
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = np.array([[sg.convolve2d(dy,w1,mode='full') for w1 in w] for dy,w in zip(self.dy,self.w)]) #padded dX, dim(k,l,m',n')
        self.dw = np.array([[sg.correlate2d(X, dy, mode='valid') for X in self.X] for dy in self.dy]) # dim(k,l,m,n)
        self.dX1 = np.sum(self.dX, axis=0)  #padded dx, dim(l,m',n')
        self.dx = np.array([pad.unpad2d(dX1, (self.m, self.n), self.mode) for dX1 in self.dX1])   #dim(l,m,n)
        return self.dx
   

    
 

 