from __future__ import print_function, division
import numpy as np
import os, shutil, math, sys
import argparse


def split_3d_volume( input_file, nbins=256, box_length=512 ):
    """
        input_file is in npz format as output of pycola
    """
    data = np.load(input_file)
    #print(len(data), type(data['px']), data['px'].dtype);

    px, py, pz = data['px'].flatten(), data['py'].flatten(), data['pz'].flatten()
    #print('px shape:',px.shape)
    
    ps = np.hstack( (px[:,None], py[:,None], pz[:,None]) );

    #print('ps shape:', ps.shape )
    #print('max :', ps.max(axis=1) )

    H, bins = np.histogramdd(ps, nbins, range=[(0,box_length)]*3 )
    #print('H', H.shape);

    count = -1
    res = [];
    for i in range(0, 256, 128):
        for j in range(0, 256, 128):
            for k in range(0, 256, 128):
                count+=1
                d = H[i:(i+128),j:(j+128),k:(k+128)]
                res.append(d);
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-i","--input_file",help="input 3D volume in npz format", type=str, required=True)
    required.add_argument("-o","--output_file",help="output 3D histograms in npz format", type=str, required=True)
    optional.add_argument("-b","--box_length",help="box length [=512]",
                        type=int, required=False, default=512)
    optional.add_argument("-n","--nbins",help="number of bins [=256]",
                        type=int, required=False, default=256)
    args = parser.parse_args()

    res = split_3d_volume(args.input_file, args.nbins, args.box_length)
    for idx, mtx in enumerate(res):
        with open( args.output_file+'_'+str(idx)+".npy", 'w') as f:
            np.save(f, mtx);
