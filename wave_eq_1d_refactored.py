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
This program implements the time dependent simulation as described in
Section 2.2.
It solves the time dependent wave equation in 1D with time and position
dependent speed of sound.

The main functions are:
    measure_spectrum_lockin - simulates the spectral measurement with lock-ins 
    measure_spectrum_fft - simulates the spectral measurement with the DAQ card
    measure_direct_lockin - simulates the interaction measurement measured with 2 lock-in
    measure_direct_fft - simulates the interaction measurement measured with one lock-in and a DAQ card

The used device can be chosen: 
    device = 'cpu' for CPU calculations
    device = 'cuda' for GPU calculations
"""

from sparse.set_float_ import float_, to_torch, DefaultFloat
import taichi as ti
float_ = DefaultFloat(ti.f64).float_
from sparse.lock_in import Lock_in_amplifier
from math import cos, sin, pi, sqrt
import numpy as np

from tqdm import tqdm
import torch

from functools import partial

from scipy.signal.windows import tukey

device = 'cpu'
if device == 'cuda':
    ti.init(arch = ti.gpu)
else:
    ti.init(arch=ti.cpu)
    
Sc = 1
#%%

@ti.func
def arr_mul_float(arr: ti.types.ndarray(float_, 3),
                  a: float_,
                  ret: ti.types.ndarray(float_, 3)):
    m, n, glob_size = arr.shape
    
    for i, j, glob_id in ti.ndrange(m, n, glob_size):
        ret[i, j, glob_id] = arr[i, j, glob_id]*a

@ti.func
def arr_add_arr(arr1: ti.types.ndarray(float_, 3),
                arr2: ti.types.ndarray(float_, 3),
                ret: ti.types.ndarray(float_, 3)):
    m, n, glob_size = arr1.shape
    
    for i, j, glob_id in ti.ndrange(m, n, glob_size):
        ret[i, j, glob_id] = arr1[i, j, glob_id] + arr2[i, j, glob_id]

@ti.func
def f(y: ti.types.ndarray(float_, 3),
      t: float_,
      c: ti.types.ndarray(float_, 2),
      h: float_,
      y_new: ti.types.ndarray(float_, 3),
      coef: float_):
    _, n, glob_size = y.shape
    
    for I in ti.grouped(ti.ndrange(n-2, glob_size)):  # grouped version
        i, glob_id = I
        i += 1  # shift to account for range starting at 1
        
        y_new[0, i, glob_id] = y[1, i, glob_id] * coef
        y_new[1, i, glob_id] = (y[0, i-1, glob_id] - 2*y[0, i, glob_id] + y[0, i+1, glob_id]) * (c[i, glob_id]/h)**2 * coef

    for glob_id in range(glob_size):  # handling boundary conditions in a simpler loop
        y_new[0, 0, glob_id] = y[1, 0, glob_id] * coef
        y_new[0, n-1, glob_id] = y[1, n-1, glob_id] * coef
        
        y_new[1, 0, glob_id] = (y[0, 0, glob_id] - 2*y[0, 1, glob_id] + y[0, 2, glob_id]) * (c[0, glob_id]/h)**2 * coef
        y_new[1, n-1, glob_id] = (y[0, n-3, glob_id] - 2*y[0, n-2, glob_id] + y[0, n-1, glob_id]) * (c[n-1, glob_id]/h)**2 * coef


@ti.kernel
def rk4_step(y: ti.types.ndarray(float_, 3), 
             t: float_, 
             dt: float_, 
             c: ti.types.ndarray(float_, 2),
             h: float_,
             y_new: ti.types.ndarray(float_, 3),
             k1: ti.types.ndarray(float_, 3),
             k2: ti.types.ndarray(float_, 3),
             k3: ti.types.ndarray(float_, 3),
             k4: ti.types.ndarray(float_, 3),
             temp1: ti.types.ndarray(float_, 3),
             temp2: ti.types.ndarray(float_, 3)):
    m, n, glob_size = y.shape
    
    # 1st step
    f(y, t, c, h, k1, dt)
    
    for I in ti.grouped(ti.ndrange(m, n, glob_size)):  # use grouped for multiple dimensions
        i, j, k = I
        temp2[i, j, k] = y[i, j, k] + 0.5 * k1[i, j, k]

    # 2nd step
    f(temp2, t + 0.5 * dt, c, h, k2, dt)
    
    for I in ti.grouped(ti.ndrange(m, n, glob_size)):
        i, j, k = I
        temp2[i, j, k] = y[i, j, k] + 0.5 * k2[i, j, k]
        
    # 3rd step
    f(temp2, t + 0.5 * dt, c, h, k3, dt)
    
    for I in ti.grouped(ti.ndrange(m, n, glob_size)):
        i, j, k = I
        temp2[i, j, k] = y[i, j, k] + k3[i, j, k]

    # 4th step
    f(temp2, t + dt, c, h, k4, dt)
    
    for I in ti.grouped(ti.ndrange(m, n, glob_size)):
        i, j, k = I
        y_new[i, j, k] = y[i, j, k] + (k1[i, j, k] + 2 * k2[i, j, k] + 2 * k3[i, j, k] + k4[i, j, k]) / 6


def source(y, ind, amp, phase, glob_id):
    y[0, ind, glob_id] += amp*sin(phase)
    
def bc(y, c0, h, R, glob_id):
    y[0, 0, glob_id] = y[0, 1, glob_id]
    y[0, -1, glob_id] = y[0, -2, glob_id] - h/c0*(1-R)/(1+R)*y[1, -1, glob_id]

def _to_array(x):
    if isinstance(x, (int, float)):
        x = np.array([x])
    elif isinstance(x, (list, np.ndarray)):
        x = np.array(x)
    else:
        raise TypeError('x needs to be int, float, list or numpy array')
    return x

def eval_spectr(sig, ref, dt, window = False):
    sig_fft = np.fft.rfft(sig)
    
    if window:
        ref_new = ref*tukey(len(ref), alpha=0.1)
    else:
        ref_new = ref
    
    
    ref_fft = np.fft.rfft(ref_new)
    freq = np.fft.rfftfreq(len(sig), dt)
    spec = sig_fft/ref_fft
    return freq, spec.real, spec.imag

def setup(a, n, l, Sc):
    h = a/(n-1)
    dt = Sc*h/c0
    
    y = torch.zeros((2, n, l), dtype=to_torch(float_)).to(device)
        
    y_new = torch.zeros_like(y)
    k1 = torch.zeros_like(y)
    k2 = torch.zeros_like(y)
    k3 = torch.zeros_like(y)
    k4 = torch.zeros_like(y)
    temp1 = torch.zeros_like(y)
    temp2 = torch.zeros_like(y)
                      
    cr = torch.ones((n, l), dtype=to_torch(float_), device=device)
    
    return h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr

def broadcast_simple(*args):
    args = [_to_array(o) for o in args]
    l = max(map(len, args))
    args = [np.broadcast_to(o, (l, )) for o in args]
    return args


def simulate(h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr, sim_t, amps, Rs, phis, c_f, c0):
    i = 0
    t = 0
    Ps = []
    with tqdm(total=1, bar_format = '{l_bar}{bar}| {n:.6f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        while t < sim_t:
            t = i*dt
            
            for k in range(len(phis)):
                source(y, 1, amps[k], phis[k](t), k)
                bc(y, c0[k], h, Rs[k], k)
            
            c = c_f(y, cr, c0)
            rk4_step(y, t, dt, c, h, y_new, k1, k2, k3, k4, temp1, temp2)
            y, y_new = y_new, y
            
            Ps.append(y[0, -1, :].cpu().clone())
            i += 1
            pbar.update(dt/sim_t)
    return np.array(Ps), np.linspace(0, t, len(Ps))

def phi_lockin(t, f):
    return 2*np.pi*f*t

def phi_fft(t, f0, f1):
    return 2*pi*(f0 + t/sim_t*(f1-f0))*t

def measure_spectrum_lockin(c0, a, R, fs, n, sim_t, amp):
    c0, R, fs, amp = broadcast_simple(c0, R, fs, amp)
    l = len(fs)
    h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr = setup(a, n, l, Sc)
    
    phis = [partial(phi_lockin, f=fs[k]) for k in range(len(fs))]
    
    Ps, ts = simulate(h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr, sim_t, amp, R, phis, lambda y, cr, c0: cr*c0, c0)
    
    xs1 = []
    ys1 = []
    
    lockin1 = Lock_in_amplifier(time_constant=0.1)
    for i in range(len(fs)):
        sig_in_FS = np.array([ts, Ps[:, i]]).T
        
        sig_out1 = lockin1.sig_out(sig_in_FS, fs[i])

        x1 = np.mean(sig_out1[0])
        y1 = np.mean(sig_out1[1])

        xs1.append(x1)
        ys1.append(y1)
    
    return fs, xs1, ys1



def measure_spectrum_fft(c0, a, R, f0, f1, n, sim_t, amp):
    c0, R, f0, f1, amp = broadcast_simple(c0, R, f0, f1, amp)
    h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr = setup(a, n, len(R), Sc)
    
    phis = [partial(phi_fft, f0=f0[k], f1=f1[k]) for k in range(len(f0))]
    
    Ps, ts = simulate(h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr, sim_t, amp, R, phis, lambda y, cr, c0: cr*c0, c0)
    
    fs, xs, ys = [], [], []
    for k, phi in enumerate(phis):
        ref = np.sin(phi(ts))
        f, x, y = eval_spectr(Ps[..., k], ref, dt)
        
        filt = (f > f0[k])*(f < f1[k])
        
        fs.append(f[filt])
        xs.append(x[filt])
        ys.append(y[filt])
    
    if len(fs) == 1:
        return fs[0], xs[0], ys[0]
    return fs, xs, ys
    

def measure_direct_fft(c0, c_SS, a, R, R_SS, fs, f_SS0, f_SS1, n, sim_t, amp1, amp2, coef):    
    l = len(fs)+1
    h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr = setup(a, n, l, Sc)
    phis = [partial(phi_lockin, f=fs[k]) for k in range(len(fs))]
    Rs = [R]*len(fs)
    amps = [amp1]*len(fs)
    c0 = [c0]*len(fs)
    
    phis.append(partial(phi_fft, f0=f_SS0, f1=f_SS1))
    Rs.append(R_SS)
    amps.append(amp2)
    c0.append(c_SS)
    
    def c_f(y, cr, c0):
        c_fluct = torch.zeros_like(cr)
        c_fluct[:, :len(fs)] = coef*y[0, :, len(fs)].reshape(n, 1) 
        c = torch.zeros_like(cr)
        c[:, :len(fs)] = c0[0]*cr[:, :len(fs)] + c_fluct[:, :len(fs)]
        c[:, len(fs)] = c_SS*cr[:, len(fs)] + c_fluct[:, len(fs)]
        return c
    
    Ps, ts = simulate(h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr, sim_t, amps, Rs, phis, c_f, c0)
    ref = np.sin(2*pi*(f_SS0+ts/sim_t*(f_SS1-f_SS0))*ts)
    
    f_SSs, x_SS, y_SS = eval_spectr(np.array(Ps)[:, -1], ref, dt)
    filt = (f_SSs > f_SS0)*(f_SSs < f_SS1)
    
    xs_FS = []
    ys_FS = []
    
    xxs = []
    xys = []
    
    yxs = []
    yys = []
    
    
    lockin1 = Lock_in_amplifier(time_constant=1e-5)
    for i in range(len(fs)):
        sig_in_FS = np.array([ts, np.array(Ps)[:, i]]).T
        
        sig_out1 = lockin1.sig_out(sig_in_FS, fs[i])

        x1 = sig_out1[0]
        y1 = sig_out1[1]

        xs_FS.append(np.mean(x1))
        ys_FS.append(np.mean(y1))
        
        _, xx, xy = eval_spectr(x1, ref, dt)
        _, yx, yy = eval_spectr(y1, ref, dt)
        
        xxs.append(xx[filt])
        xys.append(xy[filt])
        yxs.append(yx[filt])
        yys.append(yy[filt])
        
    
    return fs, f_SSs[filt], xs_FS, ys_FS, x_SS[filt], y_SS[filt], xxs, xys, yxs, yys



def measure_direct_lockin(c0, c_SS, a, R, R_SS, fs, f_SSs, n, sim_t, amp1, amp2, coef):
    X, Y = np.meshgrid(fs, f_SSs)
    
    fs = X.ravel()
    f_SSs = Y.ravel()
    f_all = np.concatenate([fs, f_SSs])
    
    l = 2*len(fs)
    Rs = np.concatenate([[R]*l, [R_SS]*l])
    amps = np.concatenate([[amp1]*l, [amp2]*l])
    c0 = np.concatenate([[c0]*l, [c_SS]*l])
    
    h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr = setup(a, n, l, Sc)
    phis = [partial(phi_lockin, f=f_all[k]) for k in range(len(f_all))]
    
    def c_f(y, cr, c0):
        c_fluct = torch.zeros_like(cr)
        c_fluct[:, :len(fs)] = coef*y[0, :, len(fs):]
        c = torch.zeros_like(cr)
        c[:, :len(fs)] = c0[0]*cr[:, :len(fs)] + c_fluct[:, :len(fs)]
        c[:, len(fs):] = c_SS*cr[:, len(fs):] + c_fluct[:, len(fs):]
        return c
    
    Ps, ts = simulate(h, dt, y, y_new, k1, k2, k3, k4, temp1, temp2, cr, sim_t, amps, Rs, phis, c_f, c0)
    
    xs1 = []
    ys1 = []
    
    xs2 = []
    ys2 = []
    
    xs3 = []
    ys3 = []
    
    lockin1 = Lock_in_amplifier(time_constant=1e-4)
    lockin2 = Lock_in_amplifier(time_constant=0.1)
    for i in tqdm(range(len(fs))):
        sig_in_FS = np.array([ts[ts > sim_t-0.1], np.array(Ps)[ts > sim_t-0.1, i]]).T
        sig_in_SS = np.array([ts[ts > sim_t-0.1], np.array(Ps)[ts > sim_t-0.1, i+len(fs)]]).T
        
        sig_out1 = lockin1.sig_out(sig_in_FS, fs[i])
        sig_out_SS = lockin2.sig_out(sig_in_SS, f_SSs[i])
        sig_out2 = lockin2.sig_out(np.array([ts[ts > sim_t-0.1], sig_out1[0]]).T, f_SSs[i])

        x1 = np.mean(sig_out1[0])
        y1 = np.mean(sig_out1[1])

        x2 = np.mean(sig_out2[0])
        y2 = np.mean(sig_out2[1])
        
        x3 = np.mean(sig_out_SS[0])
        y3 = np.mean(sig_out_SS[1])

        xs1.append(x1)
        ys1.append(y1)

        xs2.append(x2)
        ys2.append(y2)
        
        xs3.append(x3)
        ys3.append(y3)
        
    return xs1, ys1, xs2, ys2, xs3, ys3, fs, f_SSs


#%% Examples
if __name__ == '__main__':
    import time
    
    n = 200
    c0 = 300
    c_SS = 20
    a = 0.1
    R = 0.8
    R_SS = 0.8
    sim_t = 1
    f0, f1 = 10_000, 25_000
    f_SS0, f_SS1 = 1_000, 10_000
    amp2 = 0.5
    amp1 = 0.1
    
    fs = np.linspace(29_500, 30_500, 100)
    
    ret = measure_direct_fft(c0, c_SS, a, R, R_SS, fs, f_SS0, f_SS1, n, sim_t, amp1, amp2, -0.02)
    
    np.save(f'simulation_1d_acoustic_resonator_{time.strftime("%m_%d_%Y_%H_%M_%S")}.npy', 
            {'resonators': ret ,'c0': c0, 'a':a, 'R': R, 'R_SS': R_SS, 
              'c_SS': c_SS, 'freq': fs, 'f_SS0': f_SS0, 'f_SS1': f_SS1, 
              'amp1':amp1, 'amp2':amp2, 'sim_t': sim_t}) # -> this is then supposed to be plotted separately
    


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import time
    
#     n = 200
#     c0 = 300
#     c_SS = 20
#     a = 0.1
#     R = 0.8
#     sim_t = 0.1
#     f0, f1 = 10_000, 25_000
    
#     fs = [f0, f1]
#     fs = np.linspace(29_900, 30_200, 10)
    
#     ret = measure_spectrum_lockin(c0, a, R, fs, n, sim_t, 0.2)
#     ret = np.array(ret)
    
#     np.save(f'simulation_1d_acoustic_resonator_{time.strftime("%m_%d_%Y_%H_%M_%S")}.npy', 
#             {'resonators': ret ,'c0': c0, 'a':a, 'R': R, 'freq': fs})

#     fig1, ax1 = plt.subplots()
    
#     f1, x1, y1 = ret
    
#     ax1.plot(f1, x1, label = 'x')
#     ax1.plot(f1, y1, label = 'y')
#     ax1.plot(f1, abs(x1+1j*y1), label = 'r')
    
#     ax1.set_xlabel('f (Hz)')
#     ax1.set_ylabel('H(f) (a.u.)')
#     plt.grid()
#     plt.legend()