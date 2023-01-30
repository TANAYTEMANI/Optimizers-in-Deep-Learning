import numpy as np
import math
import matplotlib.pyplot as plt

def yin(x,w,b):
    return np.dot(x,w)+b

def binary_sigmoid(y_in):
    # sigmoid(x) = 1 / (1+ np.exp(-x))
    return(1/(1+np.exp(-y_in)))

def loss(x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    return(y-y_hat)**2

def dw(c,x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    derivative=y_hat*(1-y_hat)
    return(c*(y-y_hat)*derivative*x)

def db(c,x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    derivative=y_hat*(1-y_hat)
    return(c*(y-y_hat)*derivative)

def batch_gradient_descent(x,y,w,b,c,epoch):
    W=[]
    B=[]
    L=[]
    for i in range(epoch):
        cw,cb=0,0
        l=0
        for j,k in zip(x,y):
            cw+=dw(c,j,k,w,b)
            cb+=db(c,j,k,w,b)
            l+=loss(j,k,w,b)
        L.append(l/len(y))
        w+=cw/len(x)
        b+=cb/len(x)
        W.append(w)
        B.append(b)
        print("epoch:",i+1,"w =",w,"b =",b)
    return W,B,L

bw,bb,bl = batch_gradient_descent([0.5,2.5],[0.2,0.0],-2,-2,1,300)

plt.plot(bl)
plt.title('Batch Gradient Descent Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')