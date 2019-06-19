## run the workflow
from __future__ import print_function
from subprocess import Popen, PIPE
from src.gen_music_cfg import generate_cfg_file
from src.pycola_evolve import pycola_evolve
from src.hist_nbody    import split_3d_volume

## 00 generate_music_cfg file
print(" 00: generate music cfg file")
x = generate_cfg_file( './cfg/test', 0.3,-1.0,0.7) 
print(" run music")
# TODO: time
msg = Popen(['./music/MUSIC', './cfg/test.hdf5'], stdout=PIPE).communicate()[0]
print(" run pycola")
# TODO: time
pycola_evolve()
print(" run hist")





