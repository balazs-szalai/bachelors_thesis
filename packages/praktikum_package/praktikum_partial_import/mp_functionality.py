# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:10:36 2023

@author: balaz
"""

import numpy as np
import sys
import os
import time
import multiprocessing as mp
import scipy.optimize as opt
from importlib import reload
import praktikum_partial_import.mp_global_functions
from tqdm import tqdm
from multiprocessing import shared_memory
from time import sleep
import threading as thr


def part_der(f, parms, index):
    if isinstance(parms, (list)):
        eps = 1e-10
        parms0, parms1 = parms.copy(), parms.copy()
        parms0[index] += eps
        parms1[index] -= eps
        return (f(*parms0)-f(*parms1))/(2*eps)
    elif isinstance(parms, np.ndarray):
        eps = np.zeros(parms.shape)
        eps[index] = 1e-10
        return (f(*(parms+eps))-f(*(parms-eps)))/(2*eps)
    else:
        raise TypeError('parms should be a list or an np.ndarray')


def tqdm_track(n):
    try:
        shm = shared_memory.SharedMemory(create=True, size=np.array([0]*n, dtype = np.uint8).nbytes, name = 'curve_fit_state')
    except FileExistsError:
        shm = shared_memory.SharedMemory(name = 'curve_fit_state')
    shared_array = np.ndarray((n, ), dtype = np.uint8, buffer = shm.buf)
    
    s0 = 0
    with tqdm(total=n) as pbar:
        s = np.sum(shared_array)
        while s < n:
            pbar.update(s-s0)
            s0 = s
            sleep(0.1)
            s = np.sum(shared_array)
            # print(s)
        pbar.update(s-s0)

def __prenos_chyb(args):
    f, sqrt_cov, parms = args

    if isinstance(sqrt_cov, np.ndarray):
        if len(sqrt_cov.shape) == 2:
            assert sqrt_cov.shape[0] == sqrt_cov.shape[1]
            sig_2 = 0
            cov = sqrt_cov**2
            for i in range(len(cov)):
                for j in range(len(cov)):
                    sig_2 += part_der(f, parms, i)*part_der(f, parms, j)*cov[i, j]
        else:
            assert len(sqrt_cov.shape) == 1
            sig_2 = 0
            cov = sqrt_cov**2
            for i, c in enumerate(cov):
                sig_2 += part_der(f, parms, i)*part_der(f, parms, i)*c
    elif isinstance(sqrt_cov, (list, tuple)):
        sig_2 = 0
        cov = np.array(sqrt_cov)**2
        for i, c in enumerate(cov):
            sig_2 += part_der(f, parms, i)**2*c         
    else:
        raise TypeError(f'The type of sqrt_cov should be a np.ndarray, list or tuple but is {type(sqrt_cov)}')        
    return np.sqrt(sig_2)

def __sqrt_cov_type_check(sqrt_cov):
    if isinstance(sqrt_cov, np.ndarray):
        if len(sqrt_cov.shape) == 2:
            assert sqrt_cov.shape[0] == sqrt_cov.shape[1]
        else:
            assert len(sqrt_cov.shape) == 1    
    else:
        assert isinstance(sqrt_cov, (list, tuple))

def _prenos_chyb_multi(f_source, list_sqrt_cov, list_parms,number_of_threads = mp.cpu_count(), global_vars = {}, imports = None, global_function_sources = None):
    if isinstance(list_sqrt_cov, (list, tuple)):
        orig = sys.modules['__main__']
        
        path = sys.modules[__name__].__file__.split('\\')[:-2]
        path.append('praktikum_partial_import')
        path = '\\'.join(path)
        
        ind = f_source.find('(')
        f_source = f_source[:4] + 'f' + f_source[ind:]
        
        source = '\n'.join(['import ' + module.__name__ + ' as ' + name + '\n' for name, module in imports.items()])
        for func_source in global_function_sources:
            source += f'{func_source}\n'
        for key, value in global_vars.items():
            source += f'{key} = {value}\n'
        source += f_source
        source += '\n#===================================================='
        
        with open(os.path.join(path, 'mp_global_functions.py'), 'w') as f:
            f.write(source)
        import praktikum_partial_import.mp_global_functions as glob_func
        
        try:
            sys.modules['__main__'] = glob_func
            with mp.Pool(number_of_threads) as p:
                res = p.map(__prenos_chyb, [[glob_func.f, sqrt_cov, parms] for sqrt_cov, parms in zip(list_sqrt_cov, list_parms)])
        finally:
            sys.modules['__main__'] = orig
            del glob_func
        return res
    elif isinstance(list_sqrt_cov, np.ndarray):
        raise NotImplementedError('Nupmy array not implemented yet')
    else:
        raise TypeError(f'The type of sqrt_cov should be a np.ndarray, list or tuple but is {type(list_sqrt_cov)}')  
        
def __np_polyfit(data_synth):
    x, y = data_synth
    a, b = np.polyfit(x, y, 1)
    return a, b

# f = None #just to satisfy the IDE, f  is defined in exec(f_sorce)
def __opt_curve_fit(data_synth):
    xdata, ydata, p0, f, ignor_exception, n, i = data_synth
    try:
        shm = shared_memory.SharedMemory(create=True, size=np.array([0]*n, dtype = np.uint8).nbytes, name = 'curve_fit_state')
    except FileExistsError:
        shm = shared_memory.SharedMemory(name = 'curve_fit_state')
    shared_array = np.ndarray((n, ), dtype = np.uint8, buffer = shm.buf)
    try:
        parms, _ = opt.curve_fit(f, xdata, ydata, p0 = p0)
    except RuntimeError as e:
        if ignor_exception:
            return p0
        else:
            return e
    finally:
        shared_array[i] = 1
    return parms

# def random_normal(args):
#     xdata, xerr, ydata, yerr, index, gen, data_synth = args
#     for i in range(1000):
#         data_synth[1000*index+i] = gen.normal(xdata, xerr), np.random.normal(ydata, yerr)

def __random_normal(args):
    gen, n, xdata, ydata, xerr, yerr, ignor_exception = args
    res = []
    for i in range(n):
        # shared_dict[index*n+i] = gen.normal(xdata, xerr), gen.normal(ydata, yerr)
        res.append((gen.normal(xdata, xerr), gen.normal(ydata, yerr), ignor_exception))
    return res

def _lin_fit(xdata, ydata, err = None, n = 100_000, number_of_threads = 32):
    parms, cov = np.polyfit(xdata, ydata, 1, cov = True)
    a, b = parms
    s_a_stat, s_b_stat = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
    
    if isinstance(err, type(None)):
        return [(a, s_a_stat), (b, s_b_stat)]
    else:
        assert len(err) == 2
        assert len(err[0]) == len(xdata)
        assert len(err[1]) == len(ydata)
        # parms, cov = np.polyfit(xdata, ydata, 1, cov = True)
        # a, b = parms
        # s_a_stat, s_b_stat = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
        
        xerr, yerr = err
        # data_synth = [(np.random.normal(xdata, xerr), np.random.normal(ydata, yerr)) for _ in range(n)]
        # data_synth = [(np.empty(len(xdata)), np.empty(len(ydata))) for _ in range(n)]
        
        res = []
        module = sys.modules[__name__]
        orig = sys.modules['__main__']
        try:
            sys.modules['__main__'] = module
            
            ss = np.random.SeedSequence()

            child_seeds = ss.spawn(number_of_threads+1)
            streams = [np.random.default_rng(s) for s in child_seeds]
            
            nr = n//number_of_threads
            nl = n%number_of_threads
            
            with mp.Pool(number_of_threads) as p:
                args = [(gen, nr, xdata, ydata, xerr, yerr, False) for gen in streams]
                args.append((streams[-1], nl, xdata, ydata, xerr, yerr, False))
                data_synth = p.map(__random_normal, args)
                ds = [item[:-1] for sublist in data_synth for item in sublist] 
                res = p.map(__np_polyfit, ds)
                # res = p.map(np_polyfit, data_synth)
        finally:
            sys.modules['__main__'] = orig
        
        res = np.array(res)
        s_a_measure, s_b_measure = np.std(res[:, 0]), np.std(res[:, 1])
        return [(a, np.sqrt(s_a_stat**2+s_a_measure**2)), (b, np.sqrt(s_b_stat**2+ s_b_measure**2))]

def _curve_fit(f_source, xdata, ydata, err = None, p0 = None, n = 10_000, number_of_threads = 32, global_vars = None, imports = None, global_function_sources = None, ignor_exception = False):    
    path = sys.modules[__name__].__file__.split('\\')[:-2]
    path.append('praktikum_partial_import')
    path = '\\'.join(path)
    
    ind = f_source.find('(')
    f_source = f_source[:4] + 'f' + f_source[ind:]
    
    source = '\n'.join(['import ' + module.__name__ + ' as ' + name + '\n' for name, module in imports.items()])

    for key, value in global_vars.items():
        source += f'{key} = {value}\n'
    for func_source in global_function_sources:
        source += f'{func_source}\n'
    source += f_source
    source += '\n#===================================================='
    
    with open(os.path.join(path, 'mp_global_functions.py'), 'w') as f:
        f.write(source)
    glob_func = reload(praktikum_partial_import.mp_global_functions)
    
    if err == None:
        parms, cov = opt.curve_fit(glob_func.f, xdata, ydata, p0 = p0)
        s_parms = np.sqrt([cov[i, i] for i in range(len(cov))])
        return [parms, s_parms]
    else:
        assert len(err) == 2
        assert len(err[0]) == len(xdata)
        assert len(err[1]) == len(ydata)
        
        xerr, yerr = err
        
        parms, cov = opt.curve_fit(glob_func.f, xdata, ydata, p0 = p0, maxfev = 10_000)#, sigma = yerr, absolute_sigma = True)
        s_parms = np.sqrt([cov[i, i] for i in range(len(cov))])
        p0 = parms
        print(p0)
        
        # data_synth = [(np.random.normal(xdata, xerr), np.random.normal(ydata, yerr)) for _ in range(n)]
        # data_synth = [(np.empty(len(xdata)), np.empty(len(ydata))) for _ in range(n)]
        
        res = []
        
        orig = sys.modules['__main__']
        try:
            sys.modules['__main__'] = glob_func
            
            ss = np.random.SeedSequence()

            child_seeds = ss.spawn(number_of_threads+1)
            streams = [np.random.default_rng(s) for s in child_seeds]
            
            nr = n//number_of_threads
            nl = n%number_of_threads
            
            with mp.Pool(number_of_threads) as p:
                args = [(gen, nr, xdata, ydata, xerr, yerr, ignor_exception) for gen in streams[:-1]]
                args.append((streams[-1], nl, xdata, ydata, xerr, yerr, ignor_exception))
                data_synth = p.map(__random_normal, args)
            ds = []# [[item[0], item[1], p0, glob_func.f, item[2], n, i := i+1] for sublist in data_synth for item in sublist]
            j = 0
            for sublist in data_synth:
                for item in sublist:
                    ds.append([item[0], item[1], p0, glob_func.f, item[2], n, j])
                    j += 1
            t = thr.Thread(target=tqdm_track, args=(n, ))
            t.start()
            with mp.Pool(number_of_threads) as p:
                res = p.map(__opt_curve_fit, ds)
                # res = p.map(np_polyfit, data_synth)
        finally:
            sys.modules['__main__'] = orig
            try:
                t.join()
            except:
                pass
        
        for e in res:
            try:
                raise e
            except TypeError:
                pass
        res = np.array(res)
        s_measure = [np.std(res[:, i]) for i in range(len(res[0]))]
        return [parms, np.sqrt(np.array(s_parms)**2 + np.array(s_measure)**2)]