# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:25:10 2023

@author: Balázs local
"""
# import time
# t0 = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
import sys
import inspect
from praktikum_partial_import.mp_functionality import _prenos_chyb_multi, _curve_fit, _lin_fit 
from scipy.interpolate import splrep, splev
import scipy.optimize as opt
# t1 = time.perf_counter()
# print('imports: ', t1-t0)


def part_der(f, parms, index):
    '''
    Calculates the partial derivative of the function 'f' numerically at
    f(*parms) with respect to parms[index]

    Parameters
    ----------
    f : function
        The finction to de numerically derivated.
    parms : list or numpy.ndarray
        A list containing the parameters of the function 'f'.
    index : int
        The function is derivated with respect to the parameter parms[index].

    Returns
    -------
    float
        The partial derivative of the function evaluated at the given 
        parameters 'parms'.

    '''
    
    eps0 = 1e-10
    if isinstance(parms, (list)):
        parms0, parms1 = parms.copy(), parms.copy()
        parms0[index] += eps0
        parms1[index] -= eps0
        return (f(*parms0)-f(*parms1))/(2*eps0)
    elif isinstance(parms, np.ndarray):
        eps = np.zeros(parms.shape)
        eps[index] = eps0
        return (f(*(parms+eps))-f(*(parms-eps)))/(2*eps0)
    else:
        raise TypeError('parms should be a list or an np.ndarray')

def prenos_chyb(f, sqrt_cov, parms):
    '''
    Calculates the error propagated through the function 'f' from the 
    parameters 'parms' with errors 'sqrt_cov'

    Parameters
    ----------
    f : function
        The function for which we want to propagate the errors.
    sqrt_cov : list, tuple or np.ndarray
        The standard deviations of the parameters 'parms'.
    parms : list, tuple or np.ndarray
        The parameters passed into the function 'f' as f(*parms).

    Returns
    -------
    float
        The error of the value f(*parms).

    '''
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
        raise TypeError('The type of sqrt_cov should be a np.ndarray, list or tuple')        
    return np.sqrt(sig_2)

def prenos_chyb_multi(f, lists_sqrt_cov, lists_parms, number_of_threads = 1, global_vars = {}, imports = None, global_functions = None):
    '''
    Calls the function 'prenos_chyb' for each element in 'lists_sqrt_cov' and 
    'lists_parms'.
    For a long list of 'lists_parms' it is possible to use more CPU cores
    by changing the 'number_of_threads' to number bigger than 1. In this case
    all the global variables used by the function 'f' needs to be passed into
    the 'global_vars' parameter as a dictionary and the function 'f' can't 
    rely on any outside module.
    

    Parameters
    ----------
    f : function
        The function for which we want to propagate the errors.
    
    lists_sqrt_cov : list
        List of lists, tuples or np.ndarrays containing the errors of each 
        parameter we are passing into the function 'f'.
    
    lists_parms : list
        List of lists, tuples or np.ndarrays containing the values of each 
        parameter we are passing into the function 'f'.
    
    number_of_threads : int, optional
        Number of logical processors to use for the computation. This should
        only be used if the computation on 1 thread takes longer than 3 s.
        The default is 1.
    
    global_vars : dictionary, optional
        Dictionary containing the global variables by the function. Only needs 
        to be given if number_of_threads > 1. The default is {}.
    
    imports: dictionary
        A dictionary containing all the modules needed to call the function 
        'f', needs to be given as {'the_name_used_for_importing':the_actual_module_object}
    
    global_functions: list of functions
        The list of functions needed to call the function 'f'.

    Returns
    -------
    list
        List of errors of the values calculated using the function 'f' on the 
        parameters in 'lists_parms'.

    '''
    if isinstance(global_vars, type(None)):
        global_vars = {}
    if isinstance(imports, type(None)):
        imports = {}
    
    global_function_sources = []
    if not isinstance(global_functions, type(None)):
        for func in global_functions:
            global_function_sources.append(inspect.getsource(func))
    f_source = inspect.getsource(f)
    list_sqrt_cov = [[lists_sqrt_cov[i][j] for i in range(len(lists_sqrt_cov))] for j in range(len(lists_sqrt_cov[0]))]
    list_parms = [[lists_parms[i][j] for i in range(len(lists_sqrt_cov))] for j in range(len(lists_sqrt_cov[0]))]
    if number_of_threads == 1:
        return [prenos_chyb(f, sqrt_cov, parms) for sqrt_cov, parms in zip(list_sqrt_cov, list_parms)]
    else:
        return _prenos_chyb_multi(f_source, list_sqrt_cov, list_parms, number_of_threads, global_vars, imports, global_function_sources)

def round_to(x, std, latex = True):
    '''
    Rounds the value of 'x' and error of 'std' to the same number of decimal 
    places such that the value of 'std' is rounded to 1 significant value. 

    Parameters
    ----------
    x : float or int
        The vaue to be rounded.
    std : float or int
        The error of the value of 'x'.
    latex : bool, optional
        If True the returned string is formatted to latex. The default is True.

    Returns
    -------
    str
        the rounded value of 'x' ± the rounded value of 'std.

    '''
    if latex:
        if std == 0:
            return f'${np.str_(x)}$'
        dec_place = -int(np.floor(np.log10(abs(std))))
        if dec_place <= 0:
            return f'${int(round(x, dec_place))} \pm {int(round(std, dec_place))}$'
        else:
            return f'${round(x, dec_place): .{dec_place}f} \pm {round(std, dec_place): .{dec_place}f}$'
    else:
        if std == 0:
            return f'{x}'
        dec_place = -int(np.floor(np.log10(abs(std))))
        if dec_place <= 0:
            return f'{int(round(x, dec_place))} ± {int(round(std, dec_place))}'
        else:
            return f'{round(x, dec_place): .{dec_place}f} ± {round(std, dec_place): .{dec_place}f}'

markers = ['o',
           '*'	,	
           'X'	,	
           'P'	, 	
           's'	,	
           'D'	,	
           'p'	,	
           'H'	,	
           'v'	, 	
           '^'	, 	
           '<'	, 	
           '>'	, 	
           'h'	,	
           'd'	,  	
           '+'	,
           'x'	,
           '1'	, 	
           '2'	, 	
           '3'	, 	
           '4'	, 	
           '|'	,	
           '_'	,
           '.'	,	
           ','	,		
           ]


global_counter = 0
def reset_counter():
    '''
    If default_plot function is called with save = True, the plot will be
    saved as Figure_{global_counter}. This function sets global_counter = 0.
    
    Returns
    -------
    None.

    '''
    
    global global_counter
    global_counter = 0

def rand_plot_colors(n):
    '''
    Returns 'n' number of RGB colors, the first 10 are the colors normally 
    used by matplotlib.

    Parameters
    ----------
    n : int
        number of colors.

    Returns
    -------
    list[np.ndarray[3]]
        List of RGB colors as 3 long numpy arrays.

    '''
    
    colors = [np.array(c) for c in plt.cm.tab10.colors]        
    #colors = []
    if n>10:
        for i in range(n):
            colors.append(np.random.randint(10, 250, 3)/255)
        return colors
    else:
        return colors[:n]
    

def default_plot(nxdata, nydata, xlabel, ylabel, legend = None, xerror = None, yerror = None, fit = None, colors = None, marker = 'default', spline = 'lines', save=False):
    '''
    Plots nydata to nxdata. Nxdata and nydata can be 1 or more data sets to 
    plot. Can save the plot with save = True. Can plot fitting curve over the
    data using the corresponding color. Can plot error bars.

    Parameters
    ----------
    nxdata: iterable or list of iterables
        if plotting only one set of data points it should be just an iterable
        if plotting more sets of data points it should be a list of the xdata for each.
    
    nydata: same as nxdata with matching shape
        same as nxdata.
    
    xlabel: string
        label of the x axes.
    
    ylabel: string
        label of y axes.
    
    legend: string or list of strings, optional
        if plotting only one set of data it is a string with the name of the data
        if plotting more sets of data it is a list of the names of the sets of data. 
        The default is None.
    
    yerror: same as nxdata, optional
        the error corresponding to the nydata, has the same shape
        plots errorbars. The default is None.
    
    xerror: same as nxdata, optional
        the error corresponding to the nxdata, has same shape
        plots errorbars. The default is None.
    
    fit: iterable, optional
        list with each element corresponding to the set of data in nxdata, nydata
        the first element is the function you are fitting with, the others are the 
        fit parametres. The default is None.
    
    colors: string or list of strings, 3 long array or list of 3 long array, optional. 
        contains the corresponding colors to the plotted sets of data
        if fit != None, the colors has to be in form of 3 long numpy array with max value 0.6
        The default is None.
    
    marker: optional. If given, list of markers to use ('-', '--', 'o-' ect.)
             or one marker to use or one of the following: 'default', 'with lines'
    
    spline: if spline == 'spline', uses scipy to estimate a smoothing spline 
             to the data, if spline == 'lines', connects the data points with 
             lines where the width of the lines if set to the value of 'yerror'
             This is only called if fit != None.
    
    save: Bool, optional
        If True, saves the plot with the name Figure_{global_counter}.pdf. The default is False.

    Returns
    -------
    matplotlib figure, matplotlib axis (created by matplotlib.pyplot.subplots)

    '''
    # print(yerror)
    global global_counter
    global_counter += 1
    #print(global_counter)
    if isinstance(nxdata[0], (float, int)):
        nxdata = [nxdata]
    if isinstance(nydata[0], (float, int)):
        nydata = [nydata]
    if not isinstance(xerror, type(None)):
        if isinstance(xerror[0], (float, int)):
            xerror = [xerror]
    if not isinstance(yerror, type(None)):
        if isinstance(yerror[0], (float, int)):
            yerror = [yerror]
    
    for i in range(len(nxdata)):
        nxdata[i] = np.array(nxdata[i])
        nydata[i] = np.array(nydata[i])
        if not isinstance(xerror, type(None)):
            xerror[i] = np.array(xerror[i])
        if not isinstance(yerror, type(None)):
            yerror[i] = np.array(yerror[i])
    
    if marker == 'default':
        marker = markers
    elif marker == 'with lines':
        marker = ['-'+m for m in markers]
    elif marker == 'lines':
        marker = ['-' for m in markers]
    elif marker != 'default' or marker != 'with lines':
        m = marker
        marker = []
        for _ in range(len(markers)):
            marker.append(m)
    marker_len = len(marker)
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid()
    
    legend_ = legend
    if isinstance(legend, type(None)):
        if not (isinstance(nxdata[0], (float, int))):
            legend = ['']*len(nxdata)
        else:
            legend = ''
    
    if isinstance(colors, type(None)):
        colors = rand_plot_colors(len(nxdata))
    
        
    if not isinstance(xerror, type(None)) or not isinstance(yerror, type(None)):
        for i in range(len(nxdata)):
            ax1.errorbar(nxdata[i], nydata[i], yerror[i], xerror[i], marker[i%marker_len], label = legend[i], color = colors[i], capsize = 4)   
    else:
        for i in range(len(nxdata)):
            ax1.plot(nxdata[i], nydata[i], marker[i%marker_len], label = legend[i], color = colors[i])
    
    if not isinstance(fit, type(None)):
        x_max = max(max(x) for x in nxdata)
        x_min = min(min(x) for x in nxdata)
        xs = np.linspace(x_min-0.05*(x_max-x_min), x_max+0.05*(x_max-x_min), 10000)
        
        for i, xdata in enumerate(nxdata):
            ax1.plot(xs, fit[i][0](xs, *fit[i][1:]), '--', color = colors[i], alpha = 0.5)#, label = 'fit '+legend[i])
    elif spline == 'spline':    
        for i in range(len(nxdata)):
            try:
                x = np.linspace(min(nxdata[i]), max(nxdata[i]), 1000)
                spln = splrep(nxdata[i], nydata[i])
                y = splev(x, spln)
                ax1.plot(x, y, '-', alpha = 0.5)
            except Exception as e:
                print(f'no spline created for nxdata[{i}] because of the error:\n\t{e}')
    elif spline == 'lines':
        if not isinstance(yerror, type(None)):
            for i in range(len(nxdata)):
                ax1.fill_between(nxdata[i], nydata[i]+yerror[i], nydata[i]-yerror[i], color = colors[i], alpha = 0.5)
        else:
            for i in range(len(nxdata)):
                ax1.plot(nxdata[i], nydata[i], color = colors[i], alpha = 0.5)
    
    
    if not isinstance(legend_, type(None)):
        ax1.legend()
        
    if save:
        plt.savefig(f'Figure_{global_counter}.pdf', format = 'pdf')
    
    return fig1, ax1


def lin_fit(xdata, ydata, err = None, n = 100_000, number_of_threads = 32, multiprocessing = True):
    '''
    Performs a linear fit on (xdata, ydata) as y = a*x + b. If given the errors,
    performs a Monte-Carlo simulation to get the uncertainty of the fitting
    parameters caused by the uncertainties in the values of xdata and ydata.
    
    Parameters
    ----------
    
    xdata : np.ndarray
        data in the x direction
        
    ydata : np.ndarray
        data in the x direction
        
    err : 2 long itarable containinig the errors corresponding to each xdata resp. ydata, optional
        The default is None.
    
    n: int, optional
        The default is 20 000, the number of iterations for the Monte-Carlo simulation of the measurement errors
    
    number_of_threads: int, optional
        The default is 32, the number of threads to parallelise the computations on
    
    Returns
    -------
    list
        returns a list containing 2 tuple with [(a, std_a), (b, std_b)]

    '''
    if multiprocessing:
        return _lin_fit(xdata, ydata, err, n, number_of_threads)
    else:
        parms, cov = np.polyfit(xdata, ydata, 1, cov = True)
        a, b = parms
        s_a_stat, s_b_stat = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
        
        if isinstance(err, type(None)):
            return [(a, s_a_stat), (b, s_b_stat)]
        else:
            assert len(err) == 2
            assert len(err[0]) == len(xdata)
            assert len(err[1]) == len(ydata)
            
            xerr, yerr = err
            data_synth = [(np.random.normal(xdata, xerr), np.random.normal(ydata, yerr)) for _ in range(n)]
            
            res = np.array([np.polyfit(x, y, 1) for x, y, in data_synth])
            s_a_measure, s_b_measure = np.std(res[:, 0]), np.std(res[:, 1])
            return [(a, np.sqrt(s_a_stat**2+s_a_measure**2)), (b, np.sqrt(s_b_stat**2+ s_b_measure**2))]
        

def curve_fit(f, xdata, ydata, err = None, p0 = None, n = 10_000, number_of_threads = 32, global_vars = None, imports = None, global_functions = None, ignor_exception = False):
    '''
    Performs a curve fit (using scipy.optimize.curve_fit) on (xdata, ydata). 
    If given the errors, performs a Monte-Carlo simulation to get the 
    uncertainty of the fitting parameters caused by the uncertainties in the 
    values of xdata and ydata.

    Parameters
    ----------
    f : function
        the function to be fitted
        
    xdata : np.ndarray
        data in the x direction
        
    ydata : np.ndarray
        data in the y direction
        
    err : 2 long itarable
        The default is None. 2 long itarable containinig the errors corresponding to each xdata resp. ydata, optional
    
    p0 : iterable, optional
        The default is None. The initial parameters to use in the curve fitting.
    
    n: int, optional
        The default is 20 000, the number of iterations for the Monte-Carlo simulation of the measurement errors
    
    number_of_threads: int, optional
        The default is 32, the number of threads to parallelise the computations on
    
    global_vars: dictionary
        A dictionary containing all the variables which are used in the 
        function 'f' but not decleared in its local scope.
    
    imports: dictionary
        A dictionary containing all the modules needed to call the function 
        'f', needs to be given as {'the_name_used_for_importing':the_actual_module_object}
    
    global_functions: list of functions
        The list of functions needed to call the function 'f'.
    
    Returns
    -------
    list
        returns a list containing 2 np.ndarrays with [[optimal_parameters], [std_of_the_optimal_parameters]]

    '''
    if isinstance(global_vars, type(None)):
        global_vars = {}
    if isinstance(imports, type(None)):
        imports = {}
    f_source = inspect.getsource(f)
    
    global_function_sources = []
    if not isinstance(global_functions, type(None)):
        for func in global_functions:
            global_function_sources.append(inspect.getsource(func))
    return _curve_fit(f_source, xdata, ydata, err, p0, n, number_of_threads, global_vars, imports, global_function_sources, ignor_exception)

        
def readable(vals, errs = None):
    '''
    Calls the round_to function for each val, err pair from vals and errs.
    Its intended use is for writing latex tables using the function
    defualt_table.

    Parameters
    ----------
    vals : iterable
        The values to be rounded.
    errs : iterable
        The corresponding errors to the values of 'vals'

    Returns
    -------
    list
        A list countaining the values of 'vals' and 'errs' rounded with the 
        round_to function.

    '''
    
    if isinstance(errs, type(None)):
        assert len(vals) == len(errs)
        return [round_to(val, 0) for val in vals]
    return [round_to(val, err) for val, err in zip(vals, errs)]

def pad(lst, n):
    '''
    Pads the end of the list 'lst' with empty strings.
    Intended use is pad(readable(vals1, errs1), len(vals0)) where 
    len(vals0) > len(vals1) for writing latex tables with the function
    default_table
    
    '''
    
    assert n >= len(lst)
    return [lst[i] if i < len(lst) else '' for i in range(n)]
