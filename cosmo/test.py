## run the workflow
from __future__ import print_function
from subprocess import Popen, PIPE
from src.gen_music_cfg import generate_cfg_file
from src.pycola_evolve import pycola_evolve
from src.hist_nbody    import split_3d_volume
import time
import numpy as np

print(" 00: generate music cfg file")
start_time = time.time()
generate_cfg_file( './cfg/test', 0.3,-1.0,0.7) 
print("--- %s seconds ---" % (time.time() - start_time))

print(" 01: run music")
start_time = time.time()
msg = Popen(['./music/MUSIC', './cfg/test.cfg'], stdout=PIPE).communicate()[0]
print("--- %s seconds ---" % (time.time() - start_time))

print(" 02: run pycola")
start_time = time.time()
pycola_evolve('./cfg/test.hdf5', './output/test.npz', 512, 9)
print("--- %s seconds ---" % (time.time() - start_time))

print(" 03: run hist")
start_time = time.time()
res = split_3d_volume('./output/test.npz', 256, 512)
for idx, mtx in enumerate(res):
    with open( './output/test'+'_'+str(idx)+".npy", 'w') as f:
        np.save(f, mtx);
print("--- %s seconds ---" % (time.time() - start_time))





