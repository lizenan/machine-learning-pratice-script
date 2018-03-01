# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:14:17 2017

@author: Administrator
"""
import numpy as np
def para_range(n1,n2,size,rank):
    iwork1 = (n2-n1+1) / size
    iwork2 = ((n2-n1+1)% size)
    ista = rank * iwork1 + n1 + min(rank,iwork2)
    iend = ista +iwork1 - 1
    if iwork2>rank:
        iend = iend + 1
    return ista,iend

osta,oend = para_range(1,10,5,1)
local_nodes_output = int(oend - osta + 1)
who = np.random.normal(0.0, pow(200, -0.5), (local_nodes_output, 200))
record = 10
print(who.T.shape)

