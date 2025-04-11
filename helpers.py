import random
import torch

def randfloat(a,b):
    return a + (b-a)*random.random()

def tofunc(x):
    if isinstance(x,float) or isinstance(x,int): 
        return lambda : x
    if isinstance(x,slice):
        return lambda : randfloat(x.start,x.stop)
    if isinstance(x,tuple) or isinstance(x,list):
        return lambda : randfloat(x[0],x[1])
    return x

def _get_block(k,i,j):
    A=1-torch.eye(k)
    Z=torch.zeros_like(A)
    return torch.cat([Z]*i + [A] + [Z]*j,dim=1) 

def get_block_eye(k,n):
    return torch.cat([_get_block(k,i,n-i-1) for i in range(n)],dim=0)