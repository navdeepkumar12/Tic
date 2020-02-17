import numpy as np 
import pm
import tools as tl 

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
        
        t =1
        i =0
        max= np.max(np.abs(self.y)) 
        if max>10: 
            t = 1.1
            if max >50:
                t = 28
                print('overflow, max ={}, normal pass={}'.format(max,i))
            i = 0
        if max < 0.1:
            t  = 1/1.1  
            if max < 1/50:  
                t = 1/2     
                print('underflow max ={}, normal pass{}'.format(max,i))
            i = 0
        else:
            #print('normal') 
            i = i+1   
        self.y = self.y/t
        self.w = self.w/t
        self.b = self.b/t
        self.dw = self.dw/t
        self.db = self.db/t

        
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


# class sequential():
#     def __init__(self):
#         self.L = []

#     def add_layer(self,name,dim=0):
#         self.L.append([name,dim])        
