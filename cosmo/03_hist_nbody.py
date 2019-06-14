from __future__ import print_function, division
import numpy as np
import os, shutil, math, sys
import argparse

### This code will take the output of one NBody simulation (ie from the pycola code), 
### and split it into 8 sub-volumes, and histogram them.


parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_file",help="input 3D volume in npz format", type=str, required=True)
parser.add_argument("-o","--output_file",help="output 3D histograms in npz format", type=str, required=True)
parser.add_argument("-b","--box_length",help="box length", type=int, required=True)
parser.add_argument("-n","--nbins",help="number of bins", type=int, required=True)
args = parser.parse_args()



def split_3d_volume( input_file, nbins=256, box_length=512 ):
    """
        input_file is in npz format as output of pycola
    """
    data = np.load(input_file)
    print(len(data), type(data['px']), data['px'].dtype);

    px, py, pz = data['px'].flatten(), data['py'].flatten(), data['pz'].flatten()
    print('px shape:',px.shape)
    
    ps = np.hstack( (px[:,None], py[:,None], pz[:,None]) );

    print('ps shape:', ps.shape )
    print('max :', ps.max(axis=1) )

    #nbins = 8
    #box_length=128
    H, bins = np.histogramdd(ps, nbins, range=[(0,box_length)]*3 )
    print('H', H.shape);
    # for i in range(8):
    #     for j in range(8):
    #         print("i,j",i,j)
    #         print(H[i][j])
    return H

if __name__ == "__main__":
    res = split_3d_volume(args.input_file, args.nbins, args.box_length)
    np.savez(args.output_file, res)
#    np.savetxt('res.txt',res, delimiter=',')
    
#   for afile in os.listdir("OmSiNs/"):
#       if "npz" not in afile or "pycola" not in afile:
#           continue
#       infile = "OmSiNs/"+afile
#   
#       print ('Input=',infile,' of',counter)
#       counter+=1
#       if counter<start or counter>stop:
#           continue
#   
#       print (counter, infile)
#   
#       outdir = infile[:-4]
#       if os.path.exists(outdir):
#           if len(os.listdir(outdir))==8:
#               print ("done this one!")
#               continue
#           else:
#               print ('see len:',len(os.listdir(outdir)))
#               ### this is a half-empty folder. delete it!
#               shutil.rmtree(outdir)
#   
#       os.mkdir(outdir)
#       
#       ### First, read in the px/py/pz from the pycola output file
#       data = np.load(infile)
#   
#       px = data['px']
#       py = data['py']
#       pz = data['pz']
#   
#       print ('pxyz: ',px[0][0][0], py[0][0][0], pz[0][0][0])
#       
#   
#   
#       #### Try using this hp.histogramdd function...
#       ### For this I need to turn the particl elists into coord lists, 
#       ### so (  (px[i][j][k], py[i][j][k], pz[i][j][k]), ....)
#       pxf = np.ndarray.flatten(px)
#       pyf = np.ndarray.flatten(py)
#       pzf = np.ndarray.flatten(pz)
#   
#       print ('a', pxf.shape)
#       print ('b', pxf[0], pyf[0], pzf[0])
#       ### so the flattening is working. Now make this into a 3d array...
#       ps = np.vstack( (pxf, pyf, pzf) ).T
#       
#       del(pxf); del(pyf); del(pzf)
#   
#       print ("one big array!", ps.shape, ps[0,:])
#   
#       
#       ## OK! Then this is indeed a big old array. Now I want to histogram it.
#       ## this step goes from a set of parcile coordinates to a histogram of particle counts 
#       nbins = 256
#       H, bins = np.histogramdd(ps, nbins, range=((0,512),(0,512),(0,512)) )
#   
#       print ("histo dshape!", H.shape,  H[0][0][0])
#       #print ('mass sum=%.3g'%np.sum(H))
#      
#   
#       ### now I have my histogram of particle density, I split it up into 8 subvolumes and write it out
#       ### note that the file structure here is required from the legacy CosmoFlow code. One dir is created for each NBody output file, then the 8 sub-volumes are named [0-7].npy inside that dir. 
#       count = -1
#       for i in range(0, 256, 128):
#           for j in range(0, 256, 128):
#               for k in range(0, 256, 128):
#                   
#                   count+=1
#                   d = H[i:(i+128),j:(j+128),k:(k+128)]
#                   filename = outdir+"/"+str(count)+".npy"
#                   print (count,'mass sum=%.3g'%np.sum(d))
#                   np.save(filename, d)
#       print ("got count :", count)
#   
#       print ("**************************")
