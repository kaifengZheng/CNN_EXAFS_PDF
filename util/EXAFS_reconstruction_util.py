import numpy as np
import pandas as pd
from larch import Group
from larch.xafs import xftf, xftr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def chi_Nick(S, k, r, f, lambda_k, delta_k, phi_k, red_k,deg):
    k = abs(k)+.0000001
    return ((deg*S*red_k) / (k * r**2)) * f * np.exp(-2*r/lambda_k) * np.sin(2*k*r + delta_k + phi_k)
def chi_gr(SO2,k,gr,r,f,lambda_k,delta_k,red_k,phi_k):
    delta_r=r[1]-r[0]
    k = abs(k)+.0000001
    chi_k_r=np.zeros([len(k),len(r)])
    for i in range(len(k)):
        for j in range(len(r)):
            # print(k[i],f[i],r[j],phi_k[i],lambda_k[i],gr[j],red_k[i],S)
            chi_k_r[i,j]=(4*np.pi*gr[j]*delta_r)*(SO2*red_k[i]*f[i]/(k[i]))*np.sin(2*k[i]*r[j]+delta_k[i]+phi_k[i])*np.exp(-2*r[j]/lambda_k[i])
            #chi_k_r[i,j]=4*np.pi*r[j]**2*gr[j]*SO2*red_k[i]/(k[i]*reff**2)*f[i]/reff*np.sin(2*k[i]*reff+delta_k[i]+phi_k[i])*np.exp(-2*r[j]/lambda_k[i])*(r[1]-r[0])
    # print(len(np.sum(chi_k_r,axis=1)))
    return np.sum(chi_k_r,axis=1)

def chi_gr_para_matrix(SO2,k,r,reff,f,lambda_k,delta_k,red_k,phi_k):
    delta_r=r[1]-r[0]
    k = abs(k)+.0000001
    chi_k_r=np.zeros([len(k),len(r)])
    for i in range(len(k)):
        for j in range(len(r)):
            chi_k_r[i,j]=(4*np.pi*delta_r)*(SO2*red_k[i]*f[i]/(k[i]))*np.sin(2*k[i]*r[j]+delta_k[i]+phi_k[i])*np.exp(-2*r[j]/lambda_k[i])
    chi_k_r=np.array(chi_k_r)
    return chi_k_r

def intpol(data, energymesh):
    """
    data format is [[energies1, mus1]...[energy_n, mus_n]]
    """
    return np.interp(energymesh, xp=data[0:,0], fp=data[0:,1])


class DataProcessor:
    def __init__(self, folder):
        self.folder = folder
        self.df_file = None
        self.r_eff = None
        self.read_pathes()

    def read_pathes(self):
        if self.folder[-1]!='/':
            self.folder=self.folder+'/'
        column_names = ['file', 'sig2', 'amp ratio', 'deg', 'nlegs', 'r effective']
        df_dat_files=pd.read_csv(self.folder+'files.dat', skiprows=10, delimiter='\\s+', names=column_names)
        self.path_files=df_dat_files

    def read_file(self,index):
        data=[]
        start_reading = False
        if self.folder[-1]!='/':
            self.folder=self.folder+'/'
        filename=self.folder+self.path_files['file'][index]
        with open(filename, 'r') as f:
            for line in f:
                if 'k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#' in line:
                    start_reading = True
                    continue
                if start_reading:
                    data.append(line.split())
        df_file = pd.DataFrame(data)
        df_file.columns = ['k', 'real[2*phc]', 'mag[feff]', 'phase[feff]', 'red factor', 'lambda', 'real[p]@#']
        df_file = df_file.apply(pd.to_numeric, errors='coerce')
        return df_file
    def extract_r_eff(self,index):
        if self.folder[-1]!='/':
            self.folder=self.folder+'/'
        filename=self.folder+self.path_files['file'][index]
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().endswith('nleg, deg, reff, rnrmav(bohr), edge'):
                    components = line.split()
                    r_eff = float(components[2])
                    deg = float(components[1])
                    return r_eff,deg

    def plot_mag(self):
        self.df_file.plot(kind='scatter', x='k', y='mag[feff]')
        plt.xlabel('k')
        plt.ylabel('mag[feff]')
        plt.title('Scatter plot of k vs mag[feff]')
        plt.show()

    def plot_phase(self):
        self.df_file.plot(kind='scatter', x='k', y='phase[feff]')
        plt.xlabel('k')
        plt.ylabel('phase[feff]')
        plt.title('Scatter plot of k vs phase[feff]')
        plt.show()

    def plot_lambda(self):
        self.df_file.plot(kind='scatter', x='k', y='lambda')
        plt.xlabel('k')
        plt.ylabel('lambda')
        plt.title('Scatter plot of k vs lambda')
        plt.show()
    def intpol(data, energymesh):
        """
        data format is [[energies1, mus1]...[energy_n, mus_n]]
        """
        return np.interp(energymesh, xp=data[0:,0], fp=data[0:,1])
    def gr2EXAFS(self,file_index,kmesh_exafs,rmesh_exafs,rmesh_ori,gr,SO2=1):

        gr_inter=interp1d(np.round(rmesh_ori,3),np.array(gr))(np.round(rmesh_exafs,3))
        df_file=self.read_file(file_index)
        k, f, lambda_k, delta_k, phi_k, red_k = df_file[['k', 'mag[feff]','lambda', 'phase[feff]', 'real[2*phc]', 'red factor']].values.transpose()       
        reff,deg = self.extract_r_eff(file_index)
        f_interpol = intpol(np.array([k,f]).T, kmesh_exafs)
        lambda_interpol = intpol(np.array([k,lambda_k]).T, kmesh_exafs)
        delta_interpol = intpol(np.array([k,delta_k]).T, kmesh_exafs)
        phi_interpol = intpol(np.array([k,phi_k]).T, kmesh_exafs)
        red_interpol = intpol(np.array([k,red_k]).T, kmesh_exafs)
        chi_exafs=chi_gr(SO2,kmesh_exafs,gr_inter,rmesh_exafs,f_interpol,lambda_interpol,delta_interpol,red_interpol,phi_interpol)
        return kmesh_exafs,chi_exafs
    def gr2WXAFS_multi_pathes(self,file_index_set,kmesh_exafs,rmesh_exafs,rmesh_ori,gr,SO2=1):
        chi_top_all = []
        for index in file_index_set:
            kmesh_exafs,chi_top_all=self.gr2EXAFS(index,kmesh_exafs,rmesh_exafs,rmesh_ori,gr,SO2=1)
            chi_top_all.append(chi_top_all)
        chi_top_all=np.array(chi_top_all)
        chi_top=np.sum(chi_top_all,axis=0)
        return kmesh_exafs,chi_top
    def print_pathes(self):
        return(self.path_files)
    



