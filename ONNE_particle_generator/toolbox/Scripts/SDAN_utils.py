#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import (pi, arange, zeros, ones, sin, cos,
                   exp, log, sqrt, where, interp, linspace)
# from numpy.fft import fft, ifft
from scipy.fftpack import fft, ifft
from scipy.special import i0 as bessel_i0
import os
import glob
import matplotlib.pyplot as plt
from monty.io import zopen


import itertools
from fastdist import fastdist
from matplotlib.ticker import StrMethodFormatter
import shutil



sqrtpi=sqrt(pi)
FT_WINDOWS = ('Kaiser-Bessel', 'Hanning', 'Parzen', 'Welch', 'Gaussian', 'Sine')
FT_WINDOWS_SHORT = tuple([a[:3].lower() for a in FT_WINDOWS])
mass_e = 9.1e-31;
hbar = 6.626e-34/(2*np.pi);
newkmesh=np.arange(3,16.05,.05)


def read_lines(filename):

    with zopen(filename, "rt") as fobject:
                f = fobject.readlines()
                lines_str = []

                for line in f:
                    lines_str.append(line.replace("\n", "").replace("\t", " "))
    return lines_str

def xftf_fast(chi, nfft=2048, kstep=0.05, **kws):
    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chi)] = chi
    return (kstep / sqrtpi) * fft(cchi)[:int(nfft/2)]



def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None,
             window='hanning', **kws):
    """
    create a Fourier transform window array.
    Parameters:
    -------------
      x:        1-d array array to build window on.
      xmin:     starting x for FT Window
      xmax:     ending x for FT Window
      dx:       tapering parameter for FT Window
      dx2:      second tapering parameter for FT Window (=dx)
      window:   name of window type
    Returns:
    ----------
    1-d window array.
    Notes:
    -------
    Valid Window names:
        hanning              cosine-squared taper
        parzen               linear taper
        welch                quadratic taper
        gaussian             Gaussian (normal) function window
        sine                 sine function window
        kaiser               Kaiser-Bessel function-derived window
    """
    if window is None:
        window = FT_WINDOWS_SHORT[0]
    nam = window.strip().lower()[:3]
    if nam not in FT_WINDOWS_SHORT:
        raise RuntimeError("invalid window name %s" % window)

    dx1 = dx
    if dx2 is None:  dx2 = dx1
    if xmin is None: xmin = min(x)
    if xmax is None: xmax = max(x)

    xstep = (x[-1] - x[0]) / (len(x)-1)
    xeps  = 1.e-4 * xstep
    x1 = max(min(x), xmin - dx1/2.0)
    x2 = xmin + dx1/2.0  + xeps
    x3 = xmax - dx2/2.0  - xeps
    x4 = min(max(x), xmax + dx2/2.0)

    if nam == 'fha':
        if dx1 < 0: dx1 = 0
        if dx2 > 1: dx2 = 1
        x2 = x1 + xeps + dx1*(xmax-xmin)/2.0
        x3 = x4 - xeps - dx2*(xmax-xmin)/2.0
    elif nam == 'gau':
        dx1 = max(dx1, xeps)

    def asint(val): return int((val+xeps)/xstep)
    i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(x)-1, i3), min(len(x)-1, i4)
    if i2 == i1: i1 = max(0, i2-1)
    if i4 == i3: i3 = max(i2, i4-1)
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2: x2 = x2+xeps
    if x3 == x4: x4 = x4+xeps
    # initial window
    fwin =  zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = ones(i3-i2)

    # now finish making window
    if nam in ('han', 'fha'):
        fwin[i1:i2+1] = sin((pi/2)*(x[i1:i2+1]-x1) / (x2-x1))**2
        fwin[i3:i4+1] = cos((pi/2)*(x[i3:i4+1]-x3) / (x4-x3))**2
    elif nam == 'par':
        fwin[i1:i2+1] =     (x[i1:i2+1]-x1) / (x2-x1)
        fwin[i3:i4+1] = 1 - (x[i3:i4+1]-x3) / (x4-x3)
    elif nam == 'wel':
        fwin[i1:i2+1] = 1 - ((x[i1:i2+1]-x2) / (x2-x1))**2
        fwin[i3:i4+1] = 1 - ((x[i3:i4+1]-x3) / (x4-x3))**2
    elif nam  in ('kai', 'bes'):
        cen  = (x4+x1)/2
        wid  = (x4-x1)/2
        arg  = 1 - (x-cen)**2 / (wid**2)
        arg[where(arg<0)] = 0
        if nam == 'bes': # 'bes' : ifeffit 1.0 implementation of kaiser-bessel
            fwin = bessel_i0(dx* sqrt(arg)) / bessel_i0(dx)
            fwin[where(x<=x1)] = 0
            fwin[where(x>=x4)] = 0
        else: # better version
            scale = max(1.e-10, bessel_i0(dx)-1)
            fwin = (bessel_i0(dx * sqrt(arg)) - 1) / scale
    elif nam == 'sin':
        fwin[i1:i4+1] = sin(pi*(x4-x[i1:i4+1]) / (x4-x1))
    elif nam == 'gau':
        cen  = (x4+x1)/2
        fwin =  exp(-(((x - cen)**2)/(2*dx1*dx1)))
    return fwin


def xftf_prep(k, chi, kmin=0, kmax=20, kweight=2, dk=1, dk2=None,
                window='kaiser', nfft=2048, kstep=0.05):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the
    ft window.
    Returns weighted chi, window function which can easily be multiplied
    and used in xftf_fast.
    """
    if dk2 is None: dk2 = dk
    npts = int(1.01 + max(k)/kstep)
    k_max = max(max(k), kmax+dk2)
    k_   = kstep * np.arange(int(1.01+k_max/kstep), dtype='float64')
    chi_ = interp(k_, k, chi)
    win  = ftwindow(k_, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window)
    return ((chi_[:npts] *k_[:npts]**kweight), win[:npts])




def intpol(data, energymesh):
    """
    data format is [[energies1, mus1]...[energy_n, mus_n]]
    """
    return np.interp(energymesh, xp=data[0:,0], fp=data[0:,1])

def correct1(data, de, s):
    z = np.asarray(list(map(lambda n: [np.sqrt(n[0]**2+(2*mass_e*de*1.6e-19/hbar**2)*10e-20),n[1]/s], data)))
    d = z[[np.imag(0)==0 for s in z]]
    ipol = intpol(d,
                 newkmesh)
    return np.asarray([newkmesh,ipol]).transpose()

def k2(data):
    k, m = data.transpose()
    return np.asarray([k,k*k*m]).transpose()


def read_lines(filename):

    with zopen(filename, "rt") as fobject:
                f = fobject.readlines()
                lines_str = []

                for line in f:
                    lines_str.append(line.replace("\n", "").replace("\t", " "))
    return lines_str

def format_neg(number):
    """
    makes the columns in the feff.inp file look nice
    """
    if number < 0:
        return format(number, '.5f')
    else: 
        return format(number, '.6f')

def raw_lines(n):
    lines_raw = []
    for line in itertools.islice(content, (n-1)*201, 201*n, 1):
        lines_raw.append(line)
    return lines_raw[9:]

def laamps_to_xyz(input):
    nums = [l.split() for l in input[0:]]
    """
    modified to work on 92 atoms
    """
    coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums[1:])[:,2:]][:92]
    coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    mask = [fastdist.euclidean(np.asarray([0,0,0]), Z)<12 for Z in coords_centerized]
    coords=[list(map(lambda n: format_neg(float(n)), entry)) for entry in coords_centerized[mask][:92]]
    return '92'+'\n'+'0 0 0\n'+'\n'.join('Pt '+' '.join(l) for l in coords)

def laamps_to_coords(input):
    nums = [l.split() for l in input[0:]]
    """
    modified to work on 92 atoms
    """
    coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums[:])[:,2:]][:92]
    coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    """
    mask can be adjusted to select atoms within a distance 
    """
    coords=[list(map(lambda n: float(format_neg(float(n))), entry)) for entry in coords_centerized]
    return coords


def laamps_to_feff(input):
    nums = [l.split() for l in input[0:]]
    """
    modified to work on 92 atoms
    """
    coords_int = [list(map(lambda n: float(n), entry)) for entry in np.asarray(nums[1:])[:,2:]][:92]
    coords_centerized = np.asarray(coords_int)-np.asarray(coords_int).mean(axis=0)
    mask = [fastdist.euclidean(np.asarray([0,0,0]), Z)<12 for Z in coords_centerized]
    coords=[list(map(lambda n: format_neg(float(n)), entry)) for entry in coords_centerized[mask][:92]]
    return 'ATOMS'+'\n'+'\n'.join(' '.join(l)+ ' 1' + ' Pt' for l in coords) +'\nEND\n'

rmeshPrime = np.arange(0,6,.01)

from fastdist import fastdist
from numpy import trapz

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

def get_stats_np_frame(coords):
    absorbers= coords
    abs_rdf = []
    for i in range(0, len(absorbers)):
        abs_rdf.append(get_rdf_abs(np.asarray(absorbers), i, rmeshPrime))
    tot_rdf=np.mean(np.asarray(abs_rdf), axis=0)

    x,y = np.asarray([l for l in np.asarray([rmeshPrime, tot_rdf]).transpose() if 2<l[0]<3.2]).transpose()
    mean = np.average(x, weights=y)
    var = np.average((x - mean)**2, weights=y)
    return [mean, var]

def bin_list(abs_el, rmesh):
    digitized =np.digitize(
    np.asarray([l for l in abs_el if 2<l<=6])
    , rmesh)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*len(rmesh)
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/0.01
    return counter

def get_rdf(abs_, other, rmesh, normalize=None):
    dm = fastdist.vector_to_matrix_distance(abs_, other, fastdist.euclidean, "euclidean")
    counts_abs_el1 = bin_list(dm, rmesh)
    if normalize==None:
        return np.asarray(counts_abs_el1)
    else:
        return np.asarray(counts_abs_el1)/normalize
    
    