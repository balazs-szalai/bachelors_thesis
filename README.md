# Bachelor's thesis code
This repository contains all supplementary code to the bachelor's thesis:

    Interaction of acoustic and thermal waves in superfluid helium: 
        the hydrodynamic analogy of the Cherenkov effect
    Author: Balázs Szalai
    Supervisor: Mgr. Emil Varga, Ph.D.
    
    Department of Low Temperature Physics, 
    Faculty of Mathematics and Physics,
    Charles University, 
    Prague
    
    April 2025

This is just supplementary material for a more simple reproducibillity of the thesis.

## Dependencies
The $packages$ folder contains the non-standard packages. Other than that all the code only depends on standard Python packages namely:
Numpy, Scipy, Sympy, Pandas, PyTorch, Taichi (requires Pyhton 3.11 or lower), tqdm, Matplotlib (these also contain all the dependencies of the $packages$, too).
All can be installed with pip:

    pip install numpy scipy matplotlib pandas taichi torch sympy tqdm

## Contents
Although the code is not well documented, each main file contains a small description in the beginning for better orientation.

The programs 1d_eigenfreq_problem_numpy.py and wave_eq_1d.py are doing numerical simulations. The programs c1.py, c2.py and speed_of_sound.py are written for extracting the speed of sound from the measured spectra
and for examining the temperature dependence of the acoustic spectra. The script draw_interaction_image.py is useful for plotting the results of the simulation from wave_eq_1d.py and the script setup.py is only used to properly add the packages to the python environment and sets the base directory for the analised data. The packages folder contains the used packages which are either custom made or not easily accessible. 

## Assumptions
The data analysing scripts expect the access to the folder ..\clean_data. This data is archived at Zenodo at [DOI 10.5281/zenodo.15292140](https://doi.org/10.5281/zenodo.15292140).
