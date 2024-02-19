
import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
from functools import partial
import scipy.constants as consts
from scipy.optimize import minimize
from .tune_Eo_helper import *

def randomness(data_X:np.array):
    #add randomness to spectra
    random_scale=np.random.uniform(0.3,1.3,size=len(data_X))
    return np.array([random_scale[i]*data_X[i] for i in range(len(data_X))])
def kdata_cut_win(kmesh,data,bound,dx=2,window='rect'):
    if window=='rect':
        return data[(data>=bound[0])&(data<=bound[1])]
    elif window=='hanning':
        fwin=window_func(kmesh,window='hanning',dx=dx,xmin=bound[0],xmax=bound[1])
        return np.multiply(fwin,data),fwin
    
def auto_tuneE0(spectrum,deltaE):
    k_MD=spectrum['k']
    keys=list(spectrum.keys())[1:]
    spectrum_recon={}
    for i in range(len(keys)):
        chi_MD=spectrum[keys[i]]
        kmesh_recon,chi_MD_recon=reconstruction(k_MD,chi_MD,deltaE)
        if i == 0:
            spectrum_recon['k']=kmesh_recon
        spectrum_recon['config_'+str(i+1)]=chi_MD_recon
    return pd.DataFrame(spectrum_recon)
        

def window_func(kmesh,window='hanning',dx=1,xmin=None,xmax=None,dx2=None):
    """
    modified from xraylarch: https://github.com/xraypy/xraylarch/blob/master/larch/xafs/xafsft.py
    """
    dx1 = dx
    if dx2 is None:  dx2 = dx1
    if xmin is None: xmin = min(kmesh)
    if xmax is None: xmax = max(kmesh)

    xstep = kmesh[1] - kmesh[0]
    xeps  = 1.e-4 * xstep
    # print(f"xeps={xeps}")
    
    x1 = max(min(kmesh), xmin - dx1/2.0)  #starting point
    # print(f"low point: {x1}")
    x2 = xmin + dx1/2.0  + xeps# highest point of left window
    x3 = xmax - dx2/2.0  - xeps#highest point of right window
    x4 = min(max(kmesh), xmax + dx2/2.0)#lowest point of right window
    # print(f"high point2: {x3}")
    # print(f"high point: {x4}")

    def asint(val): return int((val-min(kmesh)+xeps)/xstep)

    i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
    # print(i1,i2,i3,i4)
    # print(x1,x2,x3,x4)
    # print(min(kmesh),max(kmesh))
    # print(xstep)
    # print(f"i1={i1}")
    # print(f"x[i1]={kmesh[i1]}")
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(kmesh)-1, i3), min(len(kmesh)-1, i4)
    if i2 == i1: i1 = max(0, i2-1)
    if i4 == i3: i3 = max(i2, i4-1)
    x1, x2, x3, x4 = kmesh[i1], kmesh[i2], kmesh[i3], kmesh[i4]
    if x1 == x2: x2 = x2+xeps
    if x3 == x4: x4 = x4+xeps
    # initial window
    # print(f'actually low:{kmesh[i1]}')
    # print(f"actuallyn low2{kmesh[i2]}")
    # print(f'actually high{kmesh[i3]}')
    # print(f'actually high2{kmesh[i4]}')
    fwin =  np.zeros(len(kmesh))
    if i3 > i2:
        fwin[i2:i3] = np.ones(i3-i2)

    # now finish making window
    if window=='hanning':
        fwin[i1:i2+1] = np.sin((np.pi/2)*(kmesh[i1:i2+1]-x1) / (x2-x1))**2
        fwin[i3:i4+1] = np.cos((np.pi/2)*(kmesh[i3:i4+1]-x3) / (x4-x3))**2
    return fwin

def interp_sample(spectrum,gr,kmesh,rmesh,kmesh_bound,rmesh_bound,kmesh_points,rmesh_points,kweight=2,dx=2,window_func='hanning',k_win_range=[3,7]):
    """input kspace kwight:0
    output kspace kwight:2"""
    special_bois=[]
    spectrum_collect=[]
    gr_collect=[]
    del_k=kmesh[1]-kmesh[0]
    if window_func=='hanning':
        kmesh_new=np.linspace(kmesh_bound[0],kmesh_bound[1],kmesh_points)
    else:
        kmesh_new=np.linspace(kmesh_bound[0],kmesh_bound[1],kmesh_points)
    rmesh_new=np.linspace(rmesh_bound[0],rmesh_bound[1],rmesh_points)
    keys=list(spectrum.keys())
    keys=keys[1:]
    for i in range(len(keys)):
        k2data=np.array((kmesh**kweight*spectrum[keys[i]]))
        if window_func=='hanning':
            k2data,w=kdata_cut_win(kmesh,k2data,k_win_range,dx=dx,window=window_func)
        gr_rebuild=interp1d(np.round(rmesh,3),np.array(gr[0][keys[i]]))(np.round(rmesh_new,3))
        spectrum_interp=interp1d(np.round(kmesh,3),k2data)(np.round(kmesh_new,3))

        #kwighted_spectrum
        # spectrum_interp=spectrum_interp[~np.isnan(spectrum_interp)]
        spectrum_rebuild=np.array(spectrum_interp) #np.round(kmesh_new,3)*np.round(kmesh_new,3)*

        spectrum_collect.append(spectrum_rebuild)
        gr_collect.append(gr_rebuild)
        special_bois.append([np.array(spectrum_rebuild),np.array(gr_rebuild)])
    if window_func=='hanning':
        w=interp1d(np.round(kmesh,3),w)(np.round(kmesh_new,3))
        return special_bois,kmesh_new,rmesh_new,w
    else:
        return special_bois,kmesh_new,rmesh_new
def interp_sample_gr_concate(spectrum,gr,kmesh,kmesh_bound,kmesh_points,kweight=2,dx=2,window_func='hanning',k_win_range=[3,7]):
    """input kspace kwight:0
    output kspace kwight:2"""
    special_bois=[]
    spectrum_collect=[]
    # gr_collect=[]
    del_k=kmesh[1]-kmesh[0]
    if window_func=='hanning':
        kmesh_new=np.linspace(kmesh_bound[0],kmesh_bound[1],kmesh_points)
    else:
        kmesh_new=np.linspace(kmesh_bound[0],kmesh_bound[1],kmesh_points)
    #rmesh_new=np.linspace(rmesh_bound[0],rmesh_bound[1],rmesh_points)
    keys=list(spectrum.keys())
    keys=keys[1:]
    for i in range(len(keys)):
        k2data=np.array((kmesh**kweight*spectrum[keys[i]]))
        if window_func=='hanning':
            k2data,w=kdata_cut_win(kmesh,k2data,k_win_range,dx=dx,window=window_func)
        #gr_rebuild=interp1d(np.round(rmesh,3),np.array(gr[keys[i]]))(np.round(rmesh_new,3))
        spectrum_interp=interp1d(np.round(kmesh,3),k2data)(np.round(kmesh_new,3))
        #kwighted_spectrum
        # spectrum_interp=spectrum_interp[~np.isnan(spectrum_interp)]
        spectrum_rebuild=np.array(spectrum_interp) #np.round(kmesh_new,3)*np.round(kmesh_new,3)*

        spectrum_collect.append(spectrum_rebuild)
        # gr_collect.append(gr_rebuild)
        special_bois.append([np.array(spectrum_rebuild),np.array(gr[keys[i]])])
    if window_func=='hanning':
        w=interp1d(np.round(kmesh,3),w)(np.round(kmesh_new,3))
        return special_bois,kmesh_new,w
    else:
        return special_bois,kmesh_new

def drop_bad_data(spectrum,labels=[]):
    """
       kweight=2
    """
    if type(labels)!=list:
        labels=[labels]
    k=spectrum['k']
    keys=list(spectrum.keys())[1:]
    max_value2=[]
    max_value3=[]
    for i in range(0,len(keys)):
        s_k2=np.array(abs(spectrum[keys[i]]))*k**2
        max_value2.append(s_k2.max())
    for i in range(0,len(keys)):
        s_k3=np.array(abs(spectrum[keys[i]]))*k**3
        max_value3.append(s_k3.max())
    
    max_value2=np.array(max_value2)
    max_value3=np.array(max_value3)
    index_wrong2=np.where(max_value2>=2)
    index_wrong3=np.where(max_value3>=6)
    index_wrong=np.unique(np.concatenate((index_wrong2[0],index_wrong3[0])))
    # keys_spectrum_diff=[k for k in spectrum.keys() if k not in keys_gr]
    # keys_gr_diff=[k for k in keys_gr if k not in spectrum.keys()]
    print(f'number of bad data:{len(index_wrong)}')
    wrong_keys=[keys[i] for i in index_wrong]
    # for i in range(len(index_wrong)):
    #     str_name.append(f'config_{index_wrong[i]+1}')

    spectrum_copy=spectrum.copy()
    spectrum_copy.drop(columns=wrong_keys,inplace=True)
    # spectrum_copy.drop(columns=keys_spectrum_diff,inplace=True)
    
    labels_copy=labels.copy()
    for i in range(len(labels)):
        try:
            labels_copy[i].drop(columns=wrong_keys,inplace=True)
        except:
            continue
    return spectrum_copy,labels_copy

def data_concate(gr_U,gr_F,gr_F_start,gr_F_end,gr_U_start,gr_U_end,rmesh,rmesh_bound,rmesh_points):
    """
    gr_F_start,gr_F_end,gr_U_start,gr_U_end: the distance range of gr_F and gr_U
    """
    # find the non-zero region of gr_U and gr_F(starting point)
    # rmesh_F=np.linspace(gr_F_start,gr_F_end,gr_F_num)
    # rmesh_U=np.linspace(gr_U_start,gr_U_end,gr_U_num)
    # rmesh_F_0=np.where(rmesh>=gr_F_start)[0][0]
    # rmesh_F_1=np.where(rmesh<=gr_F_end)[0][-1]
    # rmesh_U_0=np.where(rmesh>=gr_U_start)[0][0]
    # rmesh_U_1=np.where(rmesh<=gr_U_end)[0][-1]
    
    keys=list(gr_U.keys())
    rmesh_new=np.linspace(rmesh_bound[0],rmesh_bound[1],rmesh_points)
    gr_F_new={}
    gr_U_new={}
    for i in range(len(keys)):
        gr_F_new[keys[i]]=interp1d(np.round(rmesh,3),np.array(gr_F[keys[i]]))(np.round(rmesh_new,3))
        gr_U_new[keys[i]]=interp1d(np.round(rmesh,3),np.array(gr_U[keys[i]]))(np.round(rmesh_new,3))
    gr_F_new=pd.DataFrame(gr_F_new)
    gr_U_new=pd.DataFrame(gr_U_new)
    # gr_min_F=[np.where(gr_F[keys[i]])[0][0] for i in range(len(keys))]
    # gr_min_U=[np.where(gr_U[keys[i]])[0][0] for i in range(len(keys))]
    gr_F_start_index=np.where(rmesh_new>=gr_F_start)[0][0]
    gr_F_end_index=np.where(rmesh_new<=gr_F_end)[0][-1]
    gr_U_start_index=np.where(rmesh_new>=gr_U_start)[0][0]
    gr_U_end_index=np.where(rmesh_new<=gr_U_end)[0][-1]
    gr_F_cut=gr_F_new.iloc[gr_F_start_index:gr_F_end_index]
    gr_U_cut=gr_U_new.iloc[gr_U_start_index:gr_U_end_index]
    
    # for i in range(len(gr_F_cut)):
    #     gr_F_cut[keys[i]]=interp1d(np.round(rmesh[rmesh_F_0:rmesh_F_1],3),gr_F_cut[keys[i]])(np.round(rmesh_F,3))
    #     gr_U_cut[keys[i]]=interp1d(np.round(rmesh[rmesh_U_0:rmesh_U_1],3),gr_U_cut[keys[i]])(np.round(rmesh_U,3))
    gr_train=pd.concat([gr_F_cut,gr_U_cut],axis=0)
    r={
        'F_index':[0,gr_F_cut.shape[0]],
        'U_index':[gr_F_cut.shape[0],gr_F_cut.shape[0]+gr_U_cut.shape[0]],
        'r_F_index':[gr_F_start_index,gr_F_end_index],
        'r_U_index':[gr_U_start_index,gr_U_end_index],
        'rmesh_new':rmesh_new,
        'total_r_points':gr_train.shape[0],
    }
    gr_train.reset_index(inplace=True,drop=True)
    return gr_train,r
def combinator_s(special_bois):

    """
    @param special_bois: the list of frame examples(300 in my case)
    """
    num_special_bois = len(special_bois)
    #special_bois=np.array(special_bois)
    #ran_i = np.random.randint(1, num_special_bois+1, 1)
    ran_i = np.random.randint(1, 5, 1)
    ran_bois = np.random.randint(0, num_special_bois, ran_i)

    weights = np.random.dirichlet(np.ones(ran_i),size=1)
    weights = weights[0]
    # print(ran_bois)
    ran_bois_get=[special_bois[i] for i in ran_bois]
    exafs = [l[0] for l in ran_bois_get]
    exafs = np.array(exafs)
    # rdf_zn = [l[2] for l in special_bois[ran_bois]]
    # rdf_zn = np.array(rdf_zn)

    rdf_F = [l[1] for l in ran_bois_get]
    rdf_F = np.array(rdf_F)

    exafs_weighted = np.average(exafs, axis=0, weights=weights)
    # rdf_zn_weighted = np.average(rdf_zn, axis=0, weights=weights)
    rdf_F_weighted = np.average(rdf_F, axis=0, weights=weights)

    return exafs_weighted, rdf_F_weighted

def generate_examples(num_examples, special_bois):
    exafs_examples = []
    rdf_examples_zn = []
    rdf_examples_F = []

    for i in range(num_examples):
        # print(special_bois)
        exafs_example, rdf_example_F = combinator_s(special_bois)
        exafs_examples.append(exafs_example)
        #rdf_examples_zn.append(rdf_example_zn)
        rdf_examples_F.append(rdf_example_F)
    exafs_examples = np.asarray(exafs_examples)
    #rdf_examples_zn = np.asarray(rdf_examples_zn)
    rdf_examples_F = np.asarray(rdf_examples_F)
    # rdf_examples_zip = []
    # for i in range(len(exafs_examples)):
    #     rdf_examples_zip.append(np.concatenate((rdf_examples_F[i])))
    # rdf_examples_zip = np.asarray(rdf_examples_zip)
    rdf_examples_zip = rdf_examples_F
    return exafs_examples, rdf_examples_zip

def data_generator(special_bois,kmesh_new,noise_level=0.1,batch_size=128):
    while True:
        x_batch_0, y_batch = generate_examples(batch_size,special_bois)
        x_batch_0=x_batch_0
        x_batch = x_batch_0.reshape(x_batch_0.shape[0], 1,x_batch_0.shape[1])
        noise = np.random.normal(loc=0, scale=noise_level, size=x_batch.shape)
        n_samples, n_features = x_batch.shape[:2]
        n_sines = 10
        sines = np.zeros((n_samples, n_features, 1))
        for i in range(n_sines):
            freq = np.random.uniform(0.1, 10)  # frequency
            amp = np.random.uniform(noise_level, noise_level)  # amplitude
            phase = np.random.uniform(0, 2*np.pi)  # phase
            sine = amp * np.sin(2*np.pi*freq*np.arange(n_features) + phase)
            
            sines[:, :, 0] += sine
            x_result = x_batch + noise + sines
            y_result = y_batch
        yield torch.from_numpy(x_result.astype(np.float32)), torch.from_numpy(y_result.astype(np.float32))

def val_generator(special_bois,kmesh_new,batch_size=128):
    while True:     
        x_batch_0, y_batch = generate_examples(batch_size,special_bois)
        x_batch_0=x_batch_0
        x_batch = x_batch_0.reshape(x_batch_0.shape[0], 1,x_batch_0.shape[1])
        yield torch.from_numpy(x_batch.astype(np.float32)),torch.from_numpy(y_batch.astype(np.float32))
def split_gr(dict_r,gr):
    rmesh=dict_r['rmesh_new']
    # zeros=np.zeros(len(dict_r['rmesh_new']))
    gr_F=gr[0:dict_r['F_index'][1]]
    gr_U=gr[dict_r['U_index'][0]:dict_r['U_index'][1]]
    gr_F=np.pad(gr_F,(dict_r['r_F_index'][0],len(rmesh)-len(gr_F)-dict_r['r_F_index'][0]))
    gr_U=np.pad(gr_U,(dict_r['r_U_index'][0],len(rmesh)-len(gr_U)-dict_r['r_U_index'][0]))
    # rmesh_start
    gr_total=gr_F+gr_U
    return gr_F,gr_U,gr_total,rmesh
