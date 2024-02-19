import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gr_atom_from_dis_norho(dis,rmesh):
    dr=rmesh[1]-rmesh[0]
    num_bin=[]
    # V=4/3*np.pi*rmesh[-1]**3
    # N=len(dis)
    r=[]
    for i in range(len(rmesh)-1):
        num=len(np.where((dis>=rmesh[i]) & (dis<rmesh[i+1]))[0])
        r_loc=np.round(rmesh[i]+dr/2,6)
        num_nor=num/((4*np.pi*(r_loc**2)*dr))
        r.append(r_loc)
        num_bin.append(np.round(num_nor,6))
    return r,num_bin

def coord_num(gr,rmesh,r1,r2,rho):
    dr=rmesh[1]-rmesh[0]
    index_start=np.where(rmesh>=r1)[0][0]
    index_end=np.where(rmesh<=r2)[0][-1]
    const=4*np.pi*rho*dr
    cn=0
    count=index_start
    for r in rmesh[index_start:index_end]:
        cn+=gr[count]*const*r**2
        count+=1
    return cn

def coord_num(gr,rmesh,r1,r2):
    dr=rmesh[1]-rmesh[0]
    index_start=np.where(rmesh>=r1)[0][0]
    index_end=np.where(rmesh<=r2)[0][-1]
    const=4*np.pi*dr
    cn=0
    count=index_start
    for r in rmesh[index_start:index_end+1]:
        cn+=gr[count]*const*r**2
        count+=1
    return cn


def radius(gr,rmesh,r1,r2):
    index_start=np.where(rmesh>=r1)[0][0]
    index_end=np.where(rmesh<=r2)[0][-1]
    dr=rmesh[1]-rmesh[0]
    radius=np.sum(4*np.pi*gr[index_start:index_end+1]*rmesh[index_start:index_end+1]**3*dr)/coord_num(gr,rmesh,r1,r2)
    return radius

def DW_factor(gr,rmesh,r1,r2):
    index_start=np.where(rmesh>=r1)[0][0]
    index_end=np.where(rmesh<=r2)[0][-1]
    dr=rmesh[1]-rmesh[0]
    DW_factor=np.sum(4*np.pi*gr[index_start:index_end+1]*(rmesh[index_start:index_end+1]-radius(gr,rmesh,r1,r2))**2*rmesh[index_start:index_end+1]*dr)/coord_num(gr,rmesh,r1,r2)
    return DW_factor
