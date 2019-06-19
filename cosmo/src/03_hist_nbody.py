import argparse
from hist_nbody import split_3d_volume

### This code will take the output of one NBody simulation (ie from the pycola code), 
### and split it into 8 sub-volumes, and histogram them.


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

if __name__ == "__main__":
    res = split_3d_volume(args.input_file, args.nbins, args.box_length)
    for idx, mtx in enumerate(res):
        with open( args.output_file+'_'+str(idx)+".npy", 'w') as f:
            np.save(f, mtx);
