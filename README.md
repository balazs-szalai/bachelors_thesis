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
Numpy, Scipy, Sympy, Pandas, PyTorch, Taichi, tqdm, Matplotlib (these also contain all the dependencies of the $packages$, too).

## Contents
Although the code is not well documented, each main file contains a small description in the beginning for better orientation.

## Assumptions
The data analysing scripts expect the access to the folder ..\clean_data. This data is archived at Zenodo at DOI 10.5281/zenodo.15292140.
