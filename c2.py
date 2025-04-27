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
Performs the procedure described in Section 4.3.1 of the thesis for the 
measurement of the speed of the second sound.
"""

import os
import setup
data_folder = os.path.join(setup.data_folder, r'speed_of_sound\SS\fsweep*')


import numpy as np
from glob import glob
from helium_old.helium_old import SVP_to_T, torr, second_sound_velocity

from multiprocessing import shared_memory

def load_data(data_folder):
    try:
        all_data = np.load(os.path.join(data_folder, 'SS_test_one_file.npy'))
    except:
        file = '*.npy'
        
        files = sorted(glob(os.path.join(data_folder, file)))
        
        all_data = []
        
        for file in tqdm(files):
            data = np.load(file, allow_pickle=True).item()
            f = data['freq (Hz)']
            x = data['x (V)']
            y = data['y (V)']
            
            if f[0] == 25000.5 and f[-1] == 47999.5 and len(f) == 45999:
                p = data['Pressure/Temperature (Torr/K)']
                if not isinstance(p, float):
                    p = p[-1]
                all_data.append([f, x, y, [p]*len(x)])
            
        all_data = np.array(all_data)
    return all_data
    
    

#%%
from speed_of_sound import c_peaks, default_plot


def func(args):
    spec, L, L_s, ind, n, index = args
    shm = shared_memory.SharedMemory(name = f'state_indicator{index}')
    shared_array = np.ndarray((n, ), dtype = np.uint8, buffer = shm.buf)
    
    c0 = c_peaks(spec, L, L_s, filt=(50, 4), mc=True, f_range=[25, 400], harm = 1)
    shared_array[ind] = 1
    return c0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    all_data = load_data(data_folder)
    
    p_avgs = np.mean(all_data[:, 3, :], axis = 1)
    T_avgs = SVP_to_T(p_avgs*torr)
    p_stds = np.std(all_data[:, 3, :], axis = 1)

    filt_use = p_stds < 0.5
    b_num = 420

    use_data = all_data[filt_use, :, :]
    del all_data
    use_reord = np.zeros_like(use_data)
    bins = np.zeros((b_num, *use_data.shape[1:]))

    hist, bin_edges = np.histogram(T_avgs[filt_use], bins = b_num)
    mask = np.digitize(T_avgs[filt_use], bin_edges[:-1])-1

    for i in tqdm(range(b_num)):
        bins[i, :, :] = np.sum(use_data[mask == i, :, :], axis = 0)/hist[i]

    del use_reord
    del mask
    
    from multiprocessing import Pool
    from time import sleep
    
    from threading import Thread
    
    def tqdm_track(n, index = 1):
        shm = shared_memory.SharedMemory(name = f'state_indicator{index}')
        shared_array = np.ndarray((n, ), dtype = np.uint8, buffer = shm.buf)
        
        s0 = 0
        with tqdm(total=100, bar_format = '{l_bar}{bar}| {n:.2f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            s = np.mean(shared_array.astype(float))*100
            while s < 100-1e-13:
                pbar.update(s-s0)
                s0 = s
                sleep(0.1)
                s = np.mean(shared_array.astype(float))*100
            pbar.update(s-s0)
    
    from random import randint
    index = randint(0, 2**31)
    
    n = len(T_avgs[filt_use])
    
    try:
        shm = shared_memory.SharedMemory(create=True, size=n, name = f'state_indicator{index}')
    except FileExistsError:
        shm = shared_memory.SharedMemory(name = f'state_indicator{index}')
    shared_array = np.ndarray((n, ), dtype = np.uint8, buffer = shm.buf)
    shared_array[:] = 0
    
    t = Thread(target=tqdm_track, args=(n, index))
    t.start()
    with Pool(32) as p:
        args = [([use_data[i, 0, :].ravel(), use_data[i, 1, :].ravel(), use_data[i, 2, :].ravel()], 0.0312, 0.0005, i, n, index) for i in range(n)]
        res = p.map(func, args)
    shared_array[:] = 1
    t.join()
    shm.close()
    shm.unlink()
    
    stride = 1
    cs = []
    css = []
    for r in res:
        c0, c_s = r
        cs.append(c0)
        css.append(c_s)
    
    mask = np.argsort(T_avgs[filt_use])
    filt = ~np.isnan(np.array(cs, dtype = float))[mask]
    
    fig, ax = default_plot([T_avgs[filt_use][mask][filt]], [np.array(cs)[mask][filt]], 'T (K)', 'c ($\\frac{m}{s}$)',
                           xerror=[[0]*len(np.array(cs)[mask][filt])], yerror=np.array(css)[mask][filt], spline = None, legend = ['measured'])
    
    Ts = np.linspace(min(T_avgs), max(T_avgs), 1000)
    ax.plot(Ts, second_sound_velocity(Ts), label = 'reference', color = 'k', zorder=3)
    plt.legend()