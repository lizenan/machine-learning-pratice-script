# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:35:02 2017

@author: Administrator
"""

import numpy as np

def cost(b):
    return (b - 4) ** 2

def slope(b):
    return 2 * (b - 4)

b = 8
for i in range(100):
    b = b - .1 * slope(b)
    print(b)


 