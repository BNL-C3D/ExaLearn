import argparse
from pycola_evolve import pycola_evolve

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

if __name__ == '__main__':
    pycola_evolve(args.input_file, args.output_file, args.box_length, args.level, args.grid_scale)

