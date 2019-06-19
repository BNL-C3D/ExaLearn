import sys
import random

def generate_cfg_file( output_file, omega_m, w0, sigma8, seed=None ):
    if not seed:
        seed = random.randrange(1<<20)
   
    output_cfg_file = output_file+".cfg"
    output_hdf5_file = output_file+".hdf5"
    
    with open(output_cfg_file, 'w') as f:
        f.write('[setup]\n')
        f.write('boxlength = 512\n')
        f.write('zstart = 0.0\n')
        f.write('levelmin = 9\n')
        f.write('levelmax = 9\n')
        f.write('overlap = 4\n')
        f.write('align_top = no\n')
        f.write('baryons = no\n')
        f.write('use_2LPT = no\n')
        f.write('use_LLA = no\n')
        f.write('periodic_TF = yes\n')
        f.write('\n')
        f.write('\n')
        f.write('[cosmology]\n')
        f.write('Omega_m = %.6f\n' % omega_m)
        f.write('Omega_L = %.6f\n' % (1-omega_m))
        f.write('w0 = %.6f\n' % w0)
        f.write('wa = 0.0\n')
        f.write('Omega_b = 0.045\n')
        f.write('H0 = 70.0\n')
        f.write('sigma_8 = %.6f\n' % sigma8)
        f.write('nspec = 0.930515\n')
        f.write('transfer = eisenstein\n')
        f.write('\n')
        f.write('[random]\n')
        f.write('seed[9] = %d\n' % seed)
        f.write('\n')
        f.write('[output]\n')
        f.write('##generic MUSIC data format (used for testing)\n')
        f.write('##requires HDF5 installation and HDF5 enabled in Makefile\n')
        f.write('format = generic\n')
        f.write('filename = %s\n' % output_hdf5_file)
        f.write('\n')
        f.write('[poisson]\n')
        f.write('fft_fine = yes\n')
        f.write('accuracy = 1e-5\n')
        f.write('pre_smooth = 3\n')
        f.write('post_smooth = 3\n')
        f.write('smoother = gs\n')
        f.write('laplace_order = 6\n')
        f.write('grad_order = 6\n')