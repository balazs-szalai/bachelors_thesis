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
Performs the procedure described in Section 4.1.2 of the thesis for the 
measurement of the speed of the first sound.
"""

import os
import setup
data_folder = os.path.join(setup.data_folder, r'speed_of_sound\FS')

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from helium_old.helium_old import SVP_to_T, first_sound_velocity, torr
from tqdm import tqdm

from speed_of_sound import c_peaks, default_plot

def load_data(data_folder):
    try:
        all_data = np.load('FS_one_file.npy')
    except:
        file = '*.npy'
        
        files = sorted(glob(os.path.join(data_folder, file)))
        
        all_data = []
        
        for file in files:
            data = np.load(file, allow_pickle=True).item()
            f = data['freq (Hz)']
            x = data['x (V)']
            y = data['y (V)']
            p = data['Pressure (Torr)']
            
            if f[0] == 120_000 and f[-1] == 159950 and len(f) == 800:
                all_data.append([f, x, y, p])
        
        all_data = np.array(all_data)
    return all_data

#%%
if __name__ == '__main__':
    all_data = load_data(data_folder)
    p_avgs = np.mean(all_data[:, 3, :], axis = 1)
    T_avgs = SVP_to_T(p_avgs*torr)
    p_stds = np.std(all_data[:, 3, :], axis = 1)

    filt_use = p_stds < 0.5
    b_num = 38

    use_data = all_data[filt_use, :, :]
    use_reord = np.zeros_like(use_data)
    bins = np.zeros((b_num, *use_data.shape[1:]))

    hist, bin_edges = np.histogram(T_avgs[filt_use], bins = b_num)
    mask = np.digitize(T_avgs[filt_use], bin_edges[:-1])-1

    for i in range(b_num):
        bins[i, :, :] = np.sum(use_data[mask == i, :, :], axis = 0)/hist[i]
    
    f = bins[0, 0, :]
    xs, ys = bins[:, 1, :], bins[:, 2, :]
    
    fig, ax = plt.subplots()
    im = ax.imshow(abs(xs+1j*ys), aspect = 'auto', extent = [f[0]/1000, f[-1]/1000, min(T_avgs), max(T_avgs)], origin = 'lower')
    ax.set_xlabel('f (kHz)')
    ax.set_ylabel('T (K)')
    cbar = fig.colorbar(im, ax = ax, pad=0.02)
    cbar.set_label("r (a. u.)")
    
    
    filt_use = p_stds < 0.2
    use_data = all_data[filt_use, :, :]
    
    cs = []
    css = []
    for i in tqdm(range(len(T_avgs[filt_use]))):
        c0, c_s = c_peaks([use_data[i, 0, :], use_data[i, 1, :], use_data[i, 2, :]], 0.04025, 0.0002, mc=True, filt=(10, 3))
        cs.append(c0)
        css.append(c_s)
    
    
    mask = np.argsort(T_avgs[filt_use])
    
    fig, ax = default_plot([T_avgs[filt_use][mask]], [np.array(cs)[mask]], 'T (K)', 'c ($\\frac{m}{s}$)',
                           xerror=[[0]*len(cs)], yerror=np.array(css)[mask], spline = None, legend = ['measured'])
    
    Ts = np.linspace(min(T_avgs), max(T_avgs), 1000)
    ax.plot(Ts, first_sound_velocity(Ts), label = 'reference', color = 'k', zorder=3)
    plt.legend()
    
    
