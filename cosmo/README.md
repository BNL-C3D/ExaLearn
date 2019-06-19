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

## Setup Conda Environment in HPC clusters

Usually HPC clusters have `anaconda2/3` module. To see if it is there, `$ module avail`. 
Then, to load the `anaconda3`: `$ module load anaconda3`

* create pycola conda environment `conda env create -f pycola_env.yml`
* `cd music; make; cd ..`
* `cd pycola; python setup.py build_ext --inplace`

## TODO

[ ] workflow
[ ] consistent dir structure
[ ] gen batch of data given cosmological parameters.

## Troubleshoot
### Install MUSIC
* If cannot find hd5.h file, edit the CONDA env in Makefile
* If cannot find fftw3, 
    - install it from scratch

```
    wget http://www.fftw.org/fftw-3.3.8.tar.gz
    ./configure --enable-openmp --enable-float
    make install
```
    - try to allocate it using `ldconfig -v | grep fftw3`
