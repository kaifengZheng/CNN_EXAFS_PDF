#!/usr/bin/env python3

from glob import glob
from ase.io.xyz import read_xyz
from ase import Atoms
from scipy.spatial.distance import cdist
from tqdm import tqdm
import toml
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
def gr_atom_from_dis(dis,rmesh):
    dis=np.sort(dis)
    dis=dis[dis<rmesh[-1]]
    dr=rmesh[1]-rmesh[0]
    num_bin=[]
    V=4/3*np.pi*rmesh[-1]**3
    N=len(dis)
    rho=N/V
    r=[]
    for i in range(len(rmesh)-1):
        num=len(np.where((dis>=rmesh[i]) & (dis<rmesh[i+1]))[0])
        r_loc=np.round(rmesh[i]+dr/2,6)
        num_nor=num/(rho*(4*np.pi*(r_loc**2)*dr))
        r.append(r_loc)
        num_bin.append(np.round(num_nor,6))
    return r,num_bin,rho

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
def average_rdf(rdf):
    rdf_avg = np.zeros(len(rdf[0]))
    for i in range(len(rdf)):
        rdf_avg += np.array(rdf[i])
    rdf_avg /= len(rdf)
    return rdf_avg

def partial_gr(filename,rmesh,center_index=0,neighbor_syn='U',method='no_rho'):
    file1=open(filename)
    atoms=read_xyz(file1,index=0)
    atoms_get=Atoms([atom for atom in atoms])
    positions=atoms_get.get_positions()
    symbol=np.array(atoms_get.get_chemical_symbols())
    neighbor_index=np.where(symbol==neighbor_syn)[0]
    neighbor_index=neighbor_index[neighbor_index!=center_index]
    neighbor_list=positions[neighbor_index]
    dis=cdist([positions[center_index]],neighbor_list)
    dis=np.sort(dis)
    rho=0
    if method=='no_rho':
        r,gr=gr_atom_from_dis_norho(dis,rmesh)
        return r, gr
    if method=='use_rho':
        r,gr,rho=gr_atom_from_dist(dis,rmesh)
        return r,gr,rho
def main():
    config=toml.load('../config.toml')
    filenames=glob('../input/*.xyz')
    rmesh=np.arange(0.5,6,0.05)
    r=np.zeros(len(rmesh)-1)
    name_dict=dict()

    for i in tqdm(range(len(filenames)),total=len(filenames)):
        str_temp=filenames[i].split('/')[2].split('_')
        str_combine=str_temp[0]+'_'+str_temp[1]
        #print(str_combine)
        if str_combine not in name_dict.keys():
            name_dict[str_combine]=[filenames[i]]
        else:
            name_dict[str_combine].append(filenames[i])
    keys=list(name_dict.keys())
    spectrum=pd.read_csv("../output_ave_test_kx.csv")
    spectrum_k=list(spectrum.keys())
    gr_dict=dict()
    gr_dict_U=dict()
    gr_dict_F=dict()
    print(keys)
    fig,ax=plt.subplots(3,1,figsize=(10,5))
    for i in tqdm(range(len(keys)),total=len(keys)):
        gr=[]
        gr_F=[]
        gr_U=[]
        if keys[i]  not in spectrum_k:
            print(f"delete {keys[i]}")
            continue
        for j in range(len(name_dict[keys[i]])):
            file1=open(name_dict[keys[i]][j])
            atoms=read_xyz(file1,index=0)
            atoms_get=Atoms([atom for atom in atoms])
            positions=atoms_get.get_positions()
            symbol=np.array(atoms_get.get_chemical_symbols())
            #print(positions[0])
            dis=cdist([positions[0]],positions,metric='euclidean')
            dis=np.sort(dis)
            r,gr_get=gr_atom_from_dis_norho(dis,rmesh)
            r,gr_get_F=partial_gr(name_dict[keys[i]][j],rmesh,center_index=0,neighbor_syn='F',method='no_rho')
            r,gr_get_U=partial_gr(name_dict[keys[i]][j],rmesh,center_index=0,neighbor_syn='U',method='no_rho')
            gr.append(gr_get)
            gr_F.append(gr_get_F)
            gr_U.append(gr_get_U)
            file1.close()
        gr_one_all=average_rdf(gr)
        gr_one_F=average_rdf(gr_F)
        gr_one_U=average_rdf(gr_U)
        gr_dict[keys[i]]=gr_one_all
        gr_dict_F[keys[i]]=gr_one_F
        gr_dict_U[keys[i]]=gr_one_U
        ax[0].plot(r,gr_one_all)
        ax[1].plot(r,gr_one_F)
        ax[2].plot(r,gr_one_U)
    ax[0].set_title('gr_all')
    ax[1].set_title("gr_F")
    ax[2].set_title("gr_U")
    plt.savefig('gr.png')
    r=np.array(r)
    np.savetxt('rmesh.txt',r)
    pd.DataFrame(gr_dict).to_csv("gr.csv")
    pd.DataFrame(gr_dict_U).to_csv('gr_U.csv')
    pd.DataFrame(gr_dict_F).to_csv("gr_F.csv")
if __name__=='__main__':
    main()




    

