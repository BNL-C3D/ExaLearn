import sys, random, argparse

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

if __name__ == "__main__":
    # omega_m + omega_L = 1.0 
    # 0.15 <= omega_m < 0.45
    # -0.8 <= w0 <= -1.2
    # 0.5 <= sigma_8 <= 1.0

    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-o", "--output_file",help="output file as MUSIC config file", type=str, required=True)
    required.add_argument("-m", "--omega_m", help="Omega-M", metavar="[0.15,0.45)", \
            type=lambda x : (0.15 <= float(x) < 0.45) and float(x) or sys.exit("Invalid omega_m"),\
            required=True)
    required.add_argument("-w", "--w0", help="w0", metavar="[-1.2, -0.8]", \
            type=lambda x : (-1.2 <= float(x) <= -0.8) and float(x) or sys.exit("Invalid w0"),\
            required=True)
    required.add_argument("-s", "--sigma8", help="Sigma8", metavar="[0.5,1.0]",\
            type=lambda x : (0.5 <= float(x) <= 1.0) and float(x) or sys.exit("Invalid sigma8"),\
            required=True)
    optional.add_argument("--rand_seed", help="random seed", type=int)
    args = parser.parse_args()

    generate_cfg_file(args.output_file,\
            args.omega_m, \
            args.w0, \
            args.sigma8, \
            args.rand_seed)


