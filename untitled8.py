# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:29:31 2017

@author: Administrator
"""

import scipy.misc
import glob

import numpy as np
import cv2
our_own_dataset = []
our_own_target = []
for image in glob.glob("F:\\hello\\ownhandwritting\\hw_?.png"): 
    haha = cv2.imread(image)
   # gray = cv2.cvtColor(haha, cv2.COLOR_BGR2GRAY)
    subimg = cv2.resize(haha,(28,28))
    cv2.imwrite(image, subimg)
    #print(haha)
