# Praktikum
## What is this?
This is a small comprehensive Python library for all the necessary tasks for Physics practicals for any physics student who is learning to do simple measurements.
Specifically recommended for physics students of Charles University in Prague, faculty MFF.
## Why the name?
The name and some of the names of the functions might seem weird for most people. This is because when I started writing it, I didn't care aboute the language I was using, and the naming comes from the Czech language. 
It was initially written to help myself with the simple data analysis tasks in a more structured and simplified way and I naver changed the names later.
## The goal of this repository
There are functions calculating the measurement error, the propagation of uncertainty, simple default plots with some additional features, converting valid Python functions into Latex and also calculating the corresponding uncertainty to Latex equation. It is also helpful to convert data into readable latex tables with a simple default formatting.
It might have problems and it might not be perfect, but it gets the job done most of the times.

## Notes about the structure
It has 2 folders due to some of the functions using multiprocessing, which is would require to import the whole package for each process, this is however very slow due to Sympy having a lot of initializations. This was solved by adding the multiprocessing parts to a separate folder which can be imported separately, therefore lowering the overhead from the multiprocessing.
All this means is that it is necessary to handle it as 2 modules: praktikum and praktikum_partial_imports. 

## Examples
Examples are provided in the $examples$ folder. They are not meant to be understood as the actual measurements they were, they are only there to showcase the usage on some data.

## Dependencies
This package only uses very common Python libraries: Numpy, Scipy, Sympy, Matplotlib and Pandas.
