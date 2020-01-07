#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import numpy.ctypeslib as npct
import ctypes  
from ctypes import *


array_2d_double = npct.ndpointer(dtype=np.double,ndim=2,flags='C')
array_1d_double = npct.ndpointer(dtype=np.double,ndim=1,flags='C')
array_int = npct.ndpointer(dtype=np.int32,ndim=0,flags='C')
ll = ctypes.cdll.LoadLibrary

Multi_lib = ll(r".\winPGBN_sampler\libMulti_Sample.dll")
Multi_lib.Multi_Sample.restype = None
Multi_lib.Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]

Crt_Multi_lib = ll(r".\winPGBN_sampler\libCrt_Multi_Sample.dll")
Crt_Multi_lib.Crt_Multi_Sample.restype = None
Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]

realmin = 1e-8
def Calculate_pj(c_j,T):  
    p_j = []
    N = c_j[1].size
    p_j.append((1-np.exp(-1))*np.ones([1,N]))
    p_j.append(1/(1+c_j[1]))
    
    for t in [i for i in range(T+1) if i>1]:   
        tmp = -np.log(np.maximum(1-p_j[t-1],realmin))
        p_j.append(tmp/(tmp+c_j[t]))
        
    return p_j
        
def Multrnd_Matrix(X_t,Phi_t,Theta_t):
    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    Xt_to_t1_t = np.zeros([K,J], order = 'C').astype('double')
    WSZS_t = np.zeros([V,K], order = 'C').astype('double')
    
    Multi_lib.Multi_Sample(X_t,Phi_t,Theta_t, WSZS_t, Xt_to_t1_t, V,K,J)

    return Xt_to_t1_t, WSZS_t     
    
def Crt_Multirnd_Matrix(Xt_to_t1_t,Phi_t1,Theta_t1):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Kt1 = Theta_t1.shape[0]
    Xt_to_t1_t1 = np.zeros([Kt1,J],order = 'C').astype('double')
    WSZS_t1 = np.zeros([Kt,Kt1],order = 'C').astype('double')
    
    Crt_Multi_lib.Crt_Multi_Sample(Xt_to_t1_t, Phi_t1,Theta_t1, WSZS_t1, Xt_to_t1_t1, Kt, Kt1 , J)
    
    return Xt_to_t1_t1 , WSZS_t1
    
def Sample_Phi(WSZS_t,Eta_t):   
    Kt = WSZS_t.shape[0]
    Kt1 = WSZS_t.shape[1]
    Phi_t_shape = WSZS_t + Eta_t
    Phi_t = np.zeros([Kt,Kt1])
    Phi_t = np.random.gamma(Phi_t_shape,1)
    
    temp=np.sum(Phi_t,axis=0)

    tempdex = np.nonzero(temp)[0]  
    Phi_t[:,tempdex] = Phi_t[:,tempdex] / temp[tempdex]
    
    Phi_t[:,np.nonzero(temp==0)[0]]  = 0

    return Phi_t
    
def Sample_Theta(Xt_to_t1_t,c_j_t1,p_j_t,shape):
    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    Theta_t = np.zeros([Kt,N])
    Theta_t_shape = Xt_to_t1_t + shape
    Theta_t[:,:] = np.random.gamma(Theta_t_shape,1) / (c_j_t1[0,:]-np.log(np.maximum(realmin,1-p_j_t[0,:])))

    return Theta_t

def ProjSimplexSpecial(Phi_tmp,Phi_old,epsilon):
    Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
    if  np.where(Phinew[:,:]<=0)[0].size >0:
        Phinew = np.maximum(epsilon,Phinew)
        Phinew = Phinew/np.maximum(realmin,Phinew.sum(0))
    return Phinew

def Reconstruct_error(X,Phi,Theta):
    return np.power(X-np.dot(Phi,Theta),2).sum()
