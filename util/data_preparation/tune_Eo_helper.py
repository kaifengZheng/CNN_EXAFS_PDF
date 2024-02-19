import numpy as np
from scipy.interpolate import interp1d
from functools import partial
import scipy.constants as consts
from scipy.optimize import minimize

KTOE = 1.e20*consts.hbar**2 / (2*consts.m_e * consts.e) # 3.8099819442818976
ETOK = 1.0/KTOE

def etok(e,deltae=0):
    """convert photo-electron energy to wavenumber"""
    return np.sqrt((e+deltae)*ETOK)
    
def ktoe(k):
    """convert photo-electron wavenumber to energy"""
    return k*k*KTOE
def tune_E0(k,deltae):
    # e_k=ktoe(k)
    # e_k_tune=e_k+deltae
    # k_tune=etok(e_k_tune)
    # k_tune=np.array(k_tune)
    return np.array(np.sqrt(k**2+deltae*ETOK))

def comparing_exp_MD(k_MD,chi_MD,k_exp,chi_exp,kweight,deltae):
    k_tune=tune_E0(k_MD,deltae)
    k_mesh=np.linspace(k_tune[0],k_tune[-1],100)
    chi_MD_inter=interp1d(k_tune,k_tune*chi_MD)(k_mesh)
    chi_exp_inter=interp1d(k_exp,k_exp*chi_exp)(k_mesh)
    loss=(1/len(chi_exp_inter))*np.sum(k_mesh**kweight*(chi_MD_inter-chi_exp_inter)**2)
    # print(loss)
    return loss

def optimize(k_MD,chi_MD,k_exp,chi_exp,kweight=2):
    func=partial(comparing_exp_MD,k_MD,chi_MD,k_exp,chi_exp,kweight)
    opt=minimize(func,0,method="CG",bounds=[(-10,10)])
    return opt
def reconstruction(k_MD,chi_MD,deltae):
    k_tune=tune_E0(k_MD,deltae)
    k_mesh=np.linspace(k_tune[0],k_tune[-1],len(k_MD))
    chi_MD_inter=interp1d(k_tune,chi_MD)(k_mesh)
    # chi_exp_inter=interp1d(k_exp,chi_exp)(k_MD)
    return k_mesh,chi_MD_inter
