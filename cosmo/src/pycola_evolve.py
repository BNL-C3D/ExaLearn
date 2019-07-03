## modified from pycola evolve function

import numpy as np
import sys, os
PYCOLA_DIR = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],r'pycola')
sys.path.append(PYCOLA_DIR)
import pyximport
pyximport.install()
from aux import boundaries
from ic import ic_2lpt,import_music_snapshot
from evolve import evolve
from cic import CICDeposit_3
from potential import initialize_density
import argparse

def pycola_evolve(input_file, output_file, box_length, level, gridscale=3) :
    # Set up the parameters from the MUSIC ic snapshot:
    # music_file="/home/yren/EXA/code/cosmoflow-sims/MUSIC/peter_ics_bl_128.hdf5"
    #music_file="/home/yren/EXA/code/cosmoflow-sims/MUSIC/peter_ics_bl128_level3.hdf5"

    # Set up according to instructions for 
    # aux.boundaries()
    
    ## boxsize=256.0 # in Mpc/h
    ## level=8
    ## level_zoom=8
    ## gridscale=3
    boxsize=float(box_length)
    # boxsize=128.0 # in Mpc/h
    level=level
    level_zoom=level
    #gridscale=3
    
    # Set up according to instructions for 
    # ic.import_music_snapshot()
    # level0='08' # should match level above
    # level1='08' # should match level_zoom above
    
#    level0='09' # should match level above
#    level1='09' # should match level_zoom above
    level0="%02d"%(level)
    level1="%02d"%(level_zoom)

    # Set how much to cut from the sides of the full box. 
    # This makes the COLA box to be of the following size in Mpc/h:
    # (2.**level-(cut_from_sides[0]+cut_from_sides[1]))/2.**level*boxsize
    
    # This is the full box. Set FULL=True in evolve() below
    cut_from_sides=[0,0]# 100Mpc/h. 
    #
    # These are the interesting cases:
    #cut_from_sides=[64,64]# 75Mpc/h
    #cut_from_sides=[128,128] # 50Mpc/h
    #cut_from_sides=[192,192]  # 25Mpc/h



    sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, \
        sz_full_zoom1, offset_from_code \
        = import_music_snapshot(input_file, \
                                boxsize,level0=level0,level1=level1)
    
    # sx_full1 = sx_full1.astype('float32')
    # sy_full1 = sy_full1.astype('float32')
    # sz_full1 = sz_full1.astype('float32')
    # sx_full_zoom1 = sx_full_zoom1.astype('float32')
    # sy_full_zoom1 = sy_full_zoom1.astype('float32')
    # sz_full_zoom1 = sz_full_zoom1.astype('float32')

    NPART_zoom=list(sx_full_zoom1.shape)


#    print "Starting 2LPT on full box."
    
    #Get bounding boxes for full box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize  \
        = boundaries(boxsize, level, level_zoom, \
                     NPART_zoom, offset_from_code, [0,0], gridscale)

    sx2_full1, sy2_full1, sz2_full1,  sx2_full_zoom1, \
        sy2_full_zoom1, sz2_full_zoom1 \
        = ic_2lpt( 
            cellsize,
            sx_full1 ,
            sy_full1 ,
            sz_full1 ,
            
            cellsize_zoom=cellsize_zoom,
            sx_zoom = sx_full_zoom1,
            sy_zoom = sy_full_zoom1,
            sz_zoom = sz_full_zoom1,
            
            boxsize=boxsize, 
            ngrid_x_lpt=ngrid_x,ngrid_y_lpt=ngrid_y,ngrid_z_lpt=ngrid_z,
                       
            offset_zoom=offset_zoom,BBox_in=BBox_in)





    #Get bounding boxes for the COLA box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize \
        = boundaries(
            boxsize, level, level_zoom, \
            NPART_zoom, offset_from_code, cut_from_sides, gridscale)

    # Trim full-box displacement fields down to COLA volume.
    sx_full       =       sx_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sy_full       =       sy_full1[BBox_out[0,0]:BBox_out[0,1],  
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sz_full       =       sz_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sx_full_zoom  =  sx_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy_full_zoom  =  sy_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz_full_zoom  =  sz_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]               
    del sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, sz_full_zoom1

    sx2_full       =       sx2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sy2_full       =       sy2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                      BBox_out[2,0]:BBox_out[2,1]]
    sz2_full       =       sz2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sx2_full_zoom  =  sx2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy2_full_zoom  =  sy2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz2_full_zoom  =  sz2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    del sx2_full1, sy2_full1, sz2_full1, sx2_full_zoom1, sy2_full_zoom1, sz2_full_zoom1


    #print "2LPT on full box is done."
    #print "Starting COLA!"



    px, py, pz, vx, vy, vz, \
        px_zoom, py_zoom, pz_zoom, vx_zoom, vy_zoom, vz_zoom \
        = evolve( 
            cellsize,
            sx_full, sy_full, sz_full, 
            sx2_full, sy2_full, sz2_full,
            FULL=True,
            
            cellsize_zoom=cellsize_zoom,
            sx_full_zoom  = sx_full_zoom , 
            sy_full_zoom  = sy_full_zoom , 
            sz_full_zoom  = sz_full_zoom ,
            sx2_full_zoom = sx2_full_zoom,
            sy2_full_zoom = sy2_full_zoom,
            sz2_full_zoom = sz2_full_zoom,
            
            offset_zoom=offset_zoom,
            BBox_in=BBox_in,
            
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            gridcellsize=gridcellsize,
            
            ngrid_x_lpt=ngrid_x,
            ngrid_y_lpt=ngrid_y,
            ngrid_z_lpt=ngrid_z,
            gridcellsize_lpt=gridcellsize,
            
            a_final=1.,
            a_initial=1./10.,
            n_steps=10,
            
            save_to_file=True,  # set this to True to output the snapshot to a file
            file_npz_out=output_file,
            )

    del vx_zoom,vy_zoom,vz_zoom
    del vx,vy,vz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-i", "--input_file",help="input hdf5 file from MUSIC", type=str, required=True)
    required.add_argument("-o", "--output_file",help="output file in npz format", type=str, required=True)
    required.add_argument("-b", "--box-length", help="box length", type=int, required=True)
    required.add_argument("-l", "--level", help="level", type=int, required=True)
    optional.add_argument("-g", "--grid-scale", default=3, help="set grid scale [=3]", type=int)
    args = parser.parse_args()

    pycola_evolve(args.input_file, args.output_file, args.box_length, 
                  args.level, args.grid_scale)

