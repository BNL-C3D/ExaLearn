## run the workflow
from __future__ import print_function
from subprocess import Popen, PIPE
from src.gen_music_cfg import generate_cfg_file
from src.pycola_evolve import pycola_evolve
from src.hist_nbody    import split_3d_volume
from src.timeit_if     import timeit_if
import time, argparse, uuid, os
import numpy as np

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
#required.add_argument("-o", "--output_file",help="output file as MUSIC config file", type=str, required=True)
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
optional.add_argument("-v", "--verbosity", action="count")
args = parser.parse_args()

def run_music( cfg_file ):
    msg = Popen( ['./music/MUSIC', cfg_file], stdout=PIPE).communicate()[0]
    return msg

generate_cfg_file   =   timeit_if(   generate_cfg_file,   threshold=args.verbosity)
run_music           =   timeit_if(   run_music,           threshold=args.verbosity)
pycola_evolve       =   timeit_if(   pycola_evolve,       threshold=args.verbosity)
split_3d_volume     =   timeit_if(   split_3d_volume,     threshold=args.verbosity)

if __name__ == '__main__':
    fname = '_'.join(["pycola", str(args.omega_m), str(args.w0),
                      str(args.sigma8), uuid.uuid4().hex.upper()])
    print( fname )
    generate_cfg_file('./cfg/'+fname,\
            args.omega_m, \
            args.w0, \
            args.sigma8, \
            args.rand_seed)
    msg = run_music('./cfg/'+fname+'.cfg')
    pycola_evolve('./cfg/'+fname+".hdf5", './output/'+fname+'.npz', 512, 9)
    res = split_3d_volume('./output/'+fname+'.npz', 256, 512)
    for idx, mtx in enumerate(res):
        with open( './output/'+fname+'_'+str(idx)+".npy", 'w') as f:
            np.save(f, mtx);
    ## clean up
    os.remove('./cfg/'+fname+'.hdf5')
    os.remove('./output/'+fname+'.npz')
