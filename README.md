# Optimizers-in-Deep-Learning
An optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rate. Thus, it helps in reducing the overall loss and improve the accuracy.

Types of Optimizers:
<ol>
  <li>Gradient Descent</li>
  <li>Stochastic Gradient Descent</li>
  <li>Momentum</li>
  <li>Mini-Batch Gradient Descent</li>
  <li>Adagrad</li>
  <li>RMSProp</li>
  <li>AdaDelta</li>
  <li>Adam</li>
</ol>

## Stochastic Gradient Descent
<p>
Stochastic Gradient Descent (SGD) is a optimization algorithm used to update the parameters of a machine learning model in order to minimize a loss function. It's called "stochastic" because instead of computing the gradients based on the entire training dataset, SGD computes the gradients based on one randomly selected training sample in each iteration. SGD has been widely used for training large scale models due to its efficiency and ease of implementation.
</p>

### Importing Libraries

```javascript I'm A tab
import numpy as np
import math
import matplotlib.pyplot as plt
```

```javascript I'm A tab
def yin(x,w,b):
    return np.dot(x,w)+b
```

### Sigmoid Function
```javascript I'm A tab
def binary_sigmoid(y_in):
    # sigmoid(x) = 1 / (1+ np.exp(-x))
    return(1/(1+np.exp(-y_in)))
```

### Loss Function
```javascript I'm A tab
def loss(x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    return(y-y_hat)**2
```

### Change in Weights
```javascript I'm A tab
def dw(c,x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    derivative=y_hat*(1-y_hat)
    return(c*(y-y_hat)*derivative*x)
```

### Change in Bias
```javascript I'm A tab
def db(c,x,y,w,b):
    y_in=yin(x,w,b)
    y_hat=binary_sigmoid(y_in)
    derivative=y_hat*(1-y_hat)
    return(c*(y-y_hat)*derivative)
```
### Stochastic Gradient Descent
```javascript I'm A tab
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
```

### Plotting Graph between Loss & Epochs
```javascript I'm A tab
epoch = [i for i in range(1,301)]
plt.plot(epoch,sl)
plt.title('Stochastic Gradient Descent Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
```
