# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:22:11 2017

@author: Administrator
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
