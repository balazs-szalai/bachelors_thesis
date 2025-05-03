# -*- coding: utf-8 -*-
"""
This program is part of the bachelor's thesis:
    Interaction of acoustic and thermal waves in superfluid helium: 
        the hydrodynamic analogy of the Cherenkov effect
    Author: Balázs Szalai
    Supervisor: Mgr. Emil Varga, Ph.D.
    
    Department of Low Temperature Physics, 
    Faculty of Mathematics and Physics,
    Charles University, 
    Prague
    
    April 2025
"""

"""
Implements the calculation of the speed of sound based on Section 4.1.2 of
the thesis. This is used as a modul for the calculation of both the
speed of first and second sound.
It also adds an additional Sav-Gol filtering to smooth out the possible 
noise in the data.
There are multiple functions implemented, the only one used is c_peaks
which does the exact procedure described in the thesis but with the added 
Sav-Gol filter.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from praktikum import prenos_chyb, lin_fit, default_plot
from scipy.signal import savgol_filter

def res_freq_fft(spec, f0 = 1000, f1 = 2000):
    f, x, y = spec
    
    df = f[1]-f[0]
    
    r = abs(x+1j*y)
    
    freq_fft = np.fft.rfftfreq(len(x), d=df)
    
    r_fft = np.fft.rfft(r)
    r_fft[1/f1 > freq_fft] = 0
    r_fft[1/f0 < freq_fft] = 0
    r_fft_psd = abs(r_fft)

    res_freq = 1/freq_fft[np.argmax(r_fft_psd)]
    return res_freq

def calc_c(res_freq, L, res_freq_s = None, L_s = None, harm = 2):
    c0 = 2*L*res_freq*harm
    if L_s:
        c0_s = prenos_chyb(lambda f0, L: 2*L*harm*f0, [res_freq_s, L_s], [res_freq, L])
        return c0, c0_s
    else:
        return c0

def c_fft(spec, L, harm = 2):
    f, x, y = spec
    
    res_freq = res_freq_fft(spec)
    
    c0 = 2*L*res_freq*harm
    
    return c0

def make_envelope(x, y, w_len = 100, stride = 50):
    xe, ye = [], []
    i = 0 
    while i + w_len < len(y):
        ind = np.argmax(y[i: i+w_len])
        ye.append(y[i+ind])
        xe.append(x[i+ind])
        i += stride
    ind = np.argmax(y[i:])
    ye.append(y[i+ind])
    xe.append(x[i+ind])
    return [np.array(xe), np.array(ye)]

def window(x, y, x0, x1):
    filt = (x > x0)*(x < x1)
    return x[filt], y[filt]

def lorenz(f, f0, A, c):
    return A * (c/2)**2/((f-f0)**2 + (c/2)**2)

def rescale(f, r, w_len = 100, stride = 50):
    upper = make_envelope(f, r, w_len, stride)
    lower = make_envelope(f, -r, w_len, stride)
    lower[1] *= -1
        
    scale = np.interp(f, *upper) - np.interp(f, *lower)
    
    r = (r - np.interp(f, *lower))/scale
    return r

def find_peaks(spec, f_min = 1000, f_max = 2000, width = 400, w_len = 100, stride = 50, filt = False, uncertainty = False):
    f0 = res_freq_fft(spec, f_min, f_max)
    f, x, y = spec
    r = abs(x+1j*y)
    if filt:
        r = savgol_filter(r, *filt)
    r = rescale(f, r, w_len, stride)
    
    x0 = f[0]
    x1 = x0 + 1.5*f0
    peaks = []
    unc = []
    errs = []
    first = True
    while x1 <= f[-1]:
        x, y = window(f, r, x0, x1)
        try:
            parms, cov = opt.curve_fit(lorenz, x, y, p0 = [x0+f0/2, 1, width])
            if not (parms[0] > x0 and parms[0] < x1):
                raise RuntimeError
            
            errs.append(np.sqrt(cov[0,0]))
            e_avg = np.mean(errs)
            e_std = np.std(errs)
            
            if first or (abs(errs[-1] - e_avg) < 2*e_std and abs(errs[-1] - e_avg) < f0/2):
                peaks.append(parms[0])
                unc.append(np.sqrt(cov[0,0]))
            else:
                peaks.append(None)
                unc.append(None)
            x0 = parms[0] + 0.5*f0
            x1 = x0 + f0
        except RuntimeError:
            peaks.append(None)
            unc.append(None)
            x0 += f0
            x1 = x0 + f0
        first = False
    if not uncertainty:
        return np.array(peaks, dtype = float)
    else:
        return np.array([peaks, unc], dtype = float)

def c_peaks(spec, L, L_s=None, harm = 2, mc = False, filt = None, f_range = [1000, 2000]):
    peaks, errs = find_peaks(spec, *f_range, filt = filt, uncertainty=True)
    
    x = np.arange(len(peaks), dtype = float)
    x = x[~np.isnan(peaks)]
    errs = errs[~np.isnan(peaks)]
    peaks = peaks[~np.isnan(peaks)]
    
    try:
        if mc:
            (p, p_std), _ = lin_fit(x, peaks, [np.zeros_like(x), errs], n=1000, multiprocessing=False)
        else:
            (p, p_std), _ = lin_fit(x, peaks)
        
        res_freq = p
        
        c0 = calc_c(res_freq, L, p_std, L_s, harm)
        return c0
    except:
        if L_s:
            return None, None
        else:
            return None
    

def c_fit(spec, L, L_s=None, harm = 2):
    f, x, y = spec
    r = abs(x+1j*y)
    r = savgol_filter(r, 10, 3)
    
    upper = make_envelope(f, r)
    lower = make_envelope(f, -r)
    lower[1] *= -1
        
    scale = np.interp(f, *upper) - np.interp(f, *lower)
    
    r = (r - np.interp(f, *lower))/scale
    f0 = res_freq_fft(spec)
    b = 0 
    
    def fit0(x, b):
        return 0.2/(np.sin(2*np.pi*(x-b)/f0)+1+0.18) -0.03
    def fit1(x, f0):
        return 0.2/(np.sin(2*np.pi*(x-b)/f0)+1+0.18) -0.03
    def fit2(x, f0, b):
        return 0.2/(np.sin(2*np.pi*(x-b)/f0)+1+0.18) -0.03
    try:
        parms, cov = opt.curve_fit(fit0, f[:200], r[:200], p0 = [0], maxfev = 10000)
        b = parms[0]
        parms, cov = opt.curve_fit(fit1, f, r, p0 = [f0], maxfev = 10000)
        f0 = parms[0]
        parms, cov = opt.curve_fit(fit2, f, (r+1)/2, p0 = [f0, b], maxfev = 10000)
        plt.plot(f, r)
        plt.pause(1)
    except:
        return None

    
    res_freq = parms[0]
    
    c0 = 2*L*res_freq*harm
    if L_s:
        res_freq_s = np.sqrt(cov[0, 0])
        c0_s = prenos_chyb(lambda f0, L: 2*L*harm*f0, [res_freq_s, L_s], [res_freq, L])
        return c0, c0_s
    return c0

    
    
