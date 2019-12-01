""" 
code taken from "https://iamtrask.github.io/2015/07/12/basic-python-network/"
I learnt about neural networks from here, but I feel this is a little complicated
as it needs you to actually compute the matricies on paper to see what is happening.

I made a simpler model that isnt full batch training, and in my opinion, its
easier to understand for a beginner who started off like me :)
Also, the reason I find this easy is because I first learnt about neural 
networks from 3blue1brown, and my model is based on that.

"""
import numpy as np

x=np.array([[1,1,1],
            [1,0,1],
            [0,0,1],
            [0,1,1]])

y=np.array([[1,1,0,0]]).T

np.random.seed(1)
# effectively, this is the neural network.
syn=2*np.random.random((3,1))-1 # mean 0


def sig(x):
    return 1/(1+np.exp(-x))

def der(x):
    return x*(1-x)

syn0 = np.array(syn)

for it in range(1000):
    tsyn0 = np.array(syn0)
    serror = []
    
    for i in range(len(x)):
        # layer 0, in this case is x[i], 
        # layer 1, is the output layer, computed as 
        # sig(x[i].syn0) for ith input.
        # layer 1 is a simple single node.
        """
        neural network in this model:
        layer 0                    | layer 1
        x[i][0]___
                  \
                   tsyn[0]
                          \
        x[i][1]----tsyn[1] ========> l1
                          /
                   tsyn[2]
                  /
        x[i][2]---
        """
        error = y[i]-sig(np.dot(x[i],syn0))
        serror.append(error[0])
        dl = error*der(sig(np.dot(x[i],syn0)))
        tsyn0 += np.array([dl[0]*x[i]]).T
        # comment following line to get full batch training.
        syn0 = tsyn0
        # the exact same model, just that back weights get
        # updated once per batch
    serror = np.array(serror)
    syn0 = tsyn0

    l0=x
    l1=sig(np.dot(l0,syn))
    error=y-l1
    dl=error*der(l1)
    syn+=np.dot(l0.T,dl)
    #print(error.T)
    #print(dl.T)
    #print(np.dot(l0.T,dl))

    print(syn0.T,syn.T)
    print(serror,error.T)

while True:
    print("enter array to predict : ")
    tp = np.array([int(x) for x in input().split()])
    print("my model , trask's model")
    print(float(sig(np.dot(tp,syn0))),float(sig(np.dot(tp,syn))))

