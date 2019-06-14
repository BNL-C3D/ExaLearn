# Workflow

## Install 

* conda env
* fftw
* MUSIC
* PyCola

## MUSIC
generate initial conditions.


* change random seed
* change cosmological parameters
    -   Omega-m [0.15, 0.45)
    -   Omega-L (:= 1-Omega-m)
    -   sigma_8 [0.5, 1.0]   
* change output file name
* output a `*.hdf5` file

## PyCola

* `02_evolve.py`, use the initial condition MUSIC generated.
    -   `python2 ./02_evolve.py -i ../MUSIC/peter_ics_bl_512.hdf5 -o ./peter_ics_bl_512.npz -b  512 -l 9 -g 3`
* `03_hist_nbody.py`, generates the 3D histogram from the particles.
    -   `python 03_hist_nbody.py -i peter_ics_bl_512.npz -o peter_ics_hist.npz -n 256 -b 512`

## TODO

[ ] workflow
[ ] consistent dir structure
[ ] gen batch of data given cosmological parameters.
