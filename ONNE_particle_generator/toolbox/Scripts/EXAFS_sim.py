#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import (pi, arange, zeros, ones, sin, cos,
                   exp, log, sqrt, where, interp, linspace)
# from numpy.fft import fft, ifft
from scipy.fftpack import fft, ifft
from scipy.special import i0 as bessel_i0
import sys
import os
import glob
import matplotlib.pyplot as plt
from monty.io import zopen


import itertools
from fastdist import fastdist
from matplotlib.ticker import StrMethodFormatter
import shutil

from fastdist import fastdist
from numpy import trapz

sys.path.append(os.path.abspath("D:\\sbu-PHD\\Ph.D. project\\gr\\Xron-master\\Xron-master\\Scripts"))
from SDAN_utils import *

rmeshPrime = np.arange(0,6,.01)

def format_neg(number):
    """
    makes the columns in the feff.inp file look nice
    """
    if number < 0:
        return format(number, '.5f')
    else: 
        return format(number, '.6f')

def bin_list_mono(abs_el, rmesh):
    digitized =np.digitize(
    np.asarray([l for l in abs_el if 0<l<=6])
    , rmesh)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*600
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/0.01
    return counter

def get_rdf_mono(coords, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    center=np.where(all_ == min(all_))
    abs_el1 = dm[center[0][0]]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)

def get_rdf_abs(coords, abs_el1, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    abs_el1 = dm[abs_el1]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)


def integrate_mono(rdf ,rmesh, rrange):
    x,y = np.asarray([l for l in np.asarray([rmesh, rdf]).transpose() if rrange[0]<l[0]<rrange[1]]).transpose()
    return trapz(y,x)

def average_distance(rdf ,rmesh, rrange):
    x,y = np.asarray([l for l in np.asarray([rmesh, rdf]).transpose() if rrange[0]<l[0]<rrange[1]]).transpose()
    return np.sum(x*y)/np.sum(y)


def make_atoms(atomslist):
    """
    takes a list of [symbols, potentials, and coordinates]. i.e. [["a","b","c"],['0','1','2'],[[0,0,0],[0,0,1],[0,1,0]]]
    """
    if len(atomslist)==3:
        symbols=atomslist[0]
        pots=atomslist[1]
        coords=atomslist[2]
        outAtoms='\n'.join('\t'.join(str(format_neg(l)) for l in coords[num])+"\t"+str(pots[num])+"\t"+str(symbols[num]) for num in range(len(symbols)))
    if len(atomslist)<3:
        print("error list not length 3")
    return outAtoms

def make_pots(type_, abs_index, number):
    pots = [type_]*number
    if type(abs_index) == bool:
        return pots
    else:
        pots[abs_index]='0'
        return pots

def make_frame_dir(main_dir, frame):
    return main_dir + "\\frame_"+str(frame)

def make_head(headerlist):
    """
    takes a list of header options i.e.
    
[['Edge','L3'],
['S02', '1.0'],
['EXAFS', '20.0'],
['RPATH', '6'],
['NLEGS','8'],
['CRITERIA', '0.0', '0.0'],
 ['SCF', '6.0', '0', '30', '0.1', '1'],
 ['EXCHANGE', '0', '0', '0'],
 ['POTENTIALS'],
 ['0', '78', 'Pt'],
 ['1', '78', 'Pt']]
 
    """
    outHead=''.join(str(l).replace(' ',"").replace("[","").replace(",","\t").replace("]","\n").replace("'","") for l in headerlist)
    return outHead

def raw_lines(content, n):
    lines_raw = []
    for line in itertools.islice(content, (n-1)*201, 201*n, 1):
        lines_raw.append(line)
    return lines_raw[9:]

def raw_lines_mod(content, n, interval):
    lines_raw = []
    for line in itertools.islice(content, (n-1)*interval, interval*n, 1):
        lines_raw.append(line)
    return lines_raw[9:]


def laamps_to_coords(input, centerize, provided_center=None):
    nums = [l.split() for l in input]
    """
    modified to work on 92 atoms
    """
    coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
    coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    """
    mask can be adjusted to select atoms within a distance 
    """
    if centerize == True:
        if isinstance(provided_center, np.ndarray)==True:
            coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in np.asarray(coords_int)-np.asarray(provided_center)]
        else:
            coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized]
        
    if centerize == False:
        coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_int]    
    
    return coords

def laamps_to_xyz(input, type_, centerize, provided_center=None):
    
    if len(input)==1:
    
        nums = [l.split() for l in input]

        if centerize == True:
            if isinstance(provided_center, np.ndarray)==True:
                coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
                coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in np.asarray(coords_int)-np.asarray(provided_center)]
            else:
                coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
                coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
                coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized]

        if centerize == False:
            coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
            coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_int]   
        
        return str(int(len(coords_int)))+'\n'+'0 0 0\n'+'\n'.join(type_+' '+' '.join(l) for l in coords)
    
    if len(input)==2:
    
        nums = [l.split() for l in input[0]]

        if centerize == True:
            if isinstance(provided_center, np.ndarray)==True:
                coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
                coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in np.asarray(coords_int)-np.asarray(provided_center)]
            else:
                coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
                coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
                coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized]

        if centerize == False:
            coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums)[:,2:]]
            coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_int]   
        
        nums2 = [l.split() for l in input[1]]

        if centerize == True:
            if isinstance(provided_center, np.ndarray)==True:
                coords_int2 = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums2)[:,2:]]
                coords2=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in np.asarray(coords_int2)-np.asarray(provided_center)]
            else:
                coords_int2 = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums2)[:,2:]]
                coords_centerized2 = np.asarray(coords_int2)-np.asarray(coords_int2).mean(axis=0)
                coords2=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized2]

        if centerize == False:
            coords_int2 = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums2)[:,2:]]
            coords2=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_int2]   
        

        return str(int(len(coords_int)+len(coords_int2)))+'\n'+'0 0 0\n'+'\n'.join(type_[0]+' '+' '.join(l) for l in [[str(s) for s in l] for l in coords])+'\n'.join(type_[1]+' '+' '.join(l) for l in [[str(s) for s in l] for l in coords2])

def coords_to_xyz(input, num_species, type_, file_out):
    
    if num_species==1:
        coords=input
        out=str(int(len(coords)))+'\n'+'0 0 0\n'+'\n'.join(type_+' '+' '.join(l) for l in [[str(s) for s in l] for l in coords])
    
    if num_species==2:
        
        coords=input[0]
        coords2=input[1]
        out=str(int(len(coords)+len(coords2)))+'\n'+'0 0 0\n'+'\n'.join(type_[0]+' '+' '.join(l) for l in [[str(s) for s in l] for l in coords])+'\n'+'\n'.join(type_[1]+' '+' '.join(l) for l in [[str(s) for s in l] for l in coords2])
    with open(file_out, "w") as f:
        f.write(out)
        
def raw_lines_type(content, frame, type_, interval):
    return [" ".join(l) for l in [l.split() for l in raw_lines_mod(content, frame, interval)] if l[1]==type_]

def get_rdf_abs(coords, abs_el1, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    abs_el1 = dm[abs_el1]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)

def get_stats_cluster_frame(content, n, interval, cluster_size):
    absorbers, spectators = laamps_to_coords_cluster(raw_lines_mod(content, n, interval), cluster_size)
    abs_rdf = []
    for i in range(0,len(absorbers)):
        abs_rdf.append(get_rdf_abs(np.asarray(absorbers + spectators), i, rmeshPrime))
    tot_rdf=np.mean(np.asarray(abs_rdf), axis=0)

    x,y = np.asarray([l for l in np.asarray([rmeshPrime,tot_rdf]).transpose() if 2<l[0]<3.3]).transpose()
    mean = np.average(x, weights=y)
    var = np.average((x - mean)**2, weights=y)
    return [mean, var]


def laamps_to_coords_cluster(input, clustersize):
    nums = [l.split() for l in input[0:]]
    """
    modified to work on bulk with cluster size
    """
    #coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums[:])[:,2:]][:92]
    coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums[:])[:,2:]]
    coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    mask = [0<=fastdist.euclidean(np.asarray([0,0,0]), Z)<=clustersize for Z in coords_centerized]
    #coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    """
    mask can be adjusted to select atoms within a distance. 
    """
    coords_abs = [list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized[mask]]
    coords_other = np.asarray([list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized[[not l for l in mask]]])
    boundary_mask = [0<=fastdist.euclidean(np.asarray([0,0,0]), Z)<=12 for Z in coords_other]
    coords_other_in_boundary = [list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_other[boundary_mask]]
    return [coords_abs, coords_other_in_boundary]


"""
nanoparticle mods
"""
def get_stats_np_frame(coords, cutoff, provide_rdf=None):
    
    if provide_rdf==None:
        absorbers= coords
        abs_rdf = []
        for i in range(0, len(absorbers)):
            abs_rdf.append(get_rdf_abs(np.asarray(absorbers), i, rmeshPrime))
        tot_rdf=np.mean(np.asarray(abs_rdf), axis=0)

        x,y = np.asarray([l for l in np.asarray([rmeshPrime, tot_rdf]).transpose() if 2<l[0]<cutoff]).transpose()
        mean = np.average(x, weights=y)
        var = np.average((x - mean)**2, weights=y)
        return [mean, var]
    else:
        absorbers= coords
        abs_rdf = []
        for i in range(0, len(absorbers)):
            abs_rdf.append(get_rdf_abs(np.asarray(absorbers), i, rmeshPrime))
        tot_rdf=np.mean(np.asarray(abs_rdf), axis=0)
        return tot_rdf
        

def bin_list(abs_el, rmesh):
    digitized =np.digitize(
    np.asarray([l for l in abs_el if np.min(rmesh)<l<=np.max(rmesh)])
    , rmesh)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*len(rmesh)
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/(rmesh[1]-rmesh[0])
    return counter

def get_rdf(abs_, other, rmesh, normalize=None):
    dm = fastdist.vector_to_matrix_distance(abs_, other, fastdist.euclidean, "euclidean")
    counts_abs_el1 = bin_list(dm, rmesh)
    if normalize==None:
        return np.asarray(counts_abs_el1)
    else:
        return np.asarray(counts_abs_el1)/normalize




"""
Reading EXAFS data files
"""
def get_temp_from_dir(file_path):
    return int(file_path.split("\\")[-1].split('_')[-2].replace('K',''))
def get_bulk_files(files_path):
    return glob.glob(files_path+"\\*\\*\\*\\*xmu.dat")
def read_chik2(file_path):
    return np.asarray([[float(s) for s in l.split()] for l in [l for l in read_lines(file_path) if l.split()[0] != '#'][1:]])[:,[0,2]]
def read_dat(file_path):
    return np.asarray([[float(s) for s in l.split()] for l in [l for l in read_lines(file_path) if l.split()[0] != '#']])[:,[0,1]]
def get_temp_from_exp(file_path):
    return file_path.split('\\')[-1].replace('.dat','')