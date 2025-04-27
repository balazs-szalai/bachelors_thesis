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
Implementation of Section 2.1 of the thesis.
Calculates the resonance frequency shift of the first sound resonance due to
the presence of second sound resonance by solving the eigenfrequency 
problem for the 1D wave equation with position dependent speed of sound.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

def create_2nd_deriavtive(n):
    diagonals = [np.ones(n-1, dtype=np.float32),
                -2 * np.ones(n, dtype=np.float32),
                np.ones(n-1, dtype=np.float32)]
    return np.diag(diagonals[0], -1) + np.diag(diagonals[1], 0) + np.diag(diagonals[2], 1)

def add_Neumann_bc(A):
    A[0, 0] = -1
    A[0, 1] = 1
    A[-1, -2] = 1
    A[-1, -1] = -1
    return A

def compile_wave_eq_with_parametric_c(a, n, c):
    h_inv = (n-1)/a
    
    x_nodes = np.linspace(0, a, n)
    
    c_mat = np.diag(c(x_nodes)**2)
    
    A = c_mat@add_Neumann_bc(create_2nd_deriavtive(n))
    return A*h_inv**2

def solve_eq(a, n, c):
    A = compile_wave_eq_with_parametric_c(a, n, c)
    
    min_om_2 = np.linalg.eigvals(A)
    
    freqs = np.sqrt(abs(min_om_2))/(2*np.pi)
    freqs.sort()
    
    return freqs

#%%

# Parameters
a = 0.05
T_amp = 100e-6
dc = -20
c0 = 230
n, m = 10, 20

N = 500

def c(x, m, a, c0, dc, T_amp):
    return c0 + T_amp*dc*np.cos(np.pi/a*m*x)

def func(args):
    c_in = lambda x: c(x, *args[2:7])
    return solve_eq(args[0], args[1], c_in), args[2]

if __name__ == '__main__':
    img0 = np.zeros((m, n))
    img1 = np.zeros((m, n))
    
    with Pool(cpu_count()) as p:
        res1 = p.map(func, [[a, N, i, a, c0, dc, T_amp] for i in range(1, m+1)])
        res0 = p.map(func, [[a, N, i, a, c0, dc, 0] for i in range(1, m+1)])
    
    for val0, val1 in zip(res0, res1):
        img0[val0[1]-1, :] = val0[0][1:n+1]
        img1[val1[1]-1, :] = val1[0][1:n+1]
    
    analitycal = np.zeros((m, n))
    df0 = T_amp*dc/(4*a)
    for i, j in zip(range(1, m+1, 2), range(n)):
        analitycal[i, j] = df0*(j+1)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img_fig = ax[0].imshow((img1 - img0) / T_amp, aspect='auto', origin='lower', 
                           extent=[0.5, n + 0.5, 0.5, m + 0.5], interpolation='nearest')
    ax[1].imshow(analitycal / T_amp, aspect='auto', origin='lower', 
                 extent=[0.5, n + 0.5, 0.5, m + 0.5], interpolation='nearest')
    cbar = fig.colorbar(img_fig, ax=ax, location='right', aspect=40)
    
    fig.suptitle('Resonance Frequency Shift [Hz/K] for $T_{amp}$ = '+f'{T_amp} K', fontsize=18)
    ax[0].set_title('Numerical calculations', fontsize=15)
    ax[1].set_title('Analytical approximations', fontsize=15)
    ax[0].set_xlabel('FS mode', fontsize=15)
    ax[1].set_xlabel('FS mode', fontsize=15)
    ax[0].set_ylabel('SS mode', fontsize=15)
    
    xticks = np.arange(1, n + 1)
    yticks = np.arange(1, m + 1)
    for axis in ax:
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
