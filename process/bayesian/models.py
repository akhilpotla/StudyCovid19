#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
from theano.tensor import *

# define model 1
def model1(theta, t, fix_params):
    
    r = theta[0]
    p = theta[1]
    C_0 = fix_params[0]

    # if r < 1.e-5:
    #     r = 0.
    
    m = 1.
    cm = 0.
    # if p > 1.e-5 and p < 1. - 1.e-5:
    #     m = 1. / (1. - p)
    #     cm = np.power(C_0, 1./m)
    m = 1. / (1. - p)
    cm = C_0 ** 1./m
        
    # C = []
    # for i in range(len(t)):
    #     if p < 1.e-5:
    #         C.append(r * t[i] + C_0)
    #     elif p > 1. - 1.e-5:
    #         C.append(C_0 * np.exp(r * t[i]))
    #     else:
    #     C.append(np.power(r * t[i] / m + cm, m))
    C = (r * t / m + cm) ** m
            
    return C


# define model 2
def model2(theta, t, fix_params):
    
    a = theta[0]
    b = theta[1]
    C_0 = fix_params[0]
    
    # C = []
    # for i in range(len(t)):
    #     C.append(a * t[i] * np.exp(-b * t[i] * t[i]) + C_0)
    C = a * t * np.exp(-b * t * t) + C_0

    return C

# define model 3
def model3(theta, t, fix_params):
    
    a = theta[0]
    b = theta[1]
    p = theta[2]
    C_0 = fix_params[0]
    T = fix_params[1]
    # if len(fix_params) > 2:
    #     p = fix_params[2]
    
    # C = []
    # for i in range(len(t)):
    #     a1 = 1. - np.power(1. - t[i]/T, p)
    #     if a1 < 1.e-8:
    #         C.append(C_0)
    #     else:
    #         e_arg = 1. - 1. / a1
    #         C.append(a * np.exp(b * e_arg) + C_0)
    a1 = 1. - (1. - t/T) ** p
    e_arg = 1. - 1. / a1
    C = a * np.exp(b * e_arg) + C_0
            
    return C
