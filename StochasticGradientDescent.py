# implement gradient descent 
import numpy as np
import math
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


def stochastic_gradient_descent(x,y,w,b,c,epoch):
    W=[]
    B=[]
    L=[]
    for i in range(epoch):
        l=0
        for j,k in zip(x,y):
            w_new=w+dw(c,j,k,w,b)
            b_new=b+db(c,j,k,w,b)
            l+=loss(j,k,w,b)
            w=w_new
            b=b_new
        L.append(l/len(y))
        W.append(w)
        B.append(b)
        print("epoch:",i+1,"w =",w,"b =",b)
    return W,B,L