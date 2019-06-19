from gen_music_cfg import generate_cfg_file
import argparse

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

if __name__ == "__main__":
    # omega_m + omega_L = 1.0 
    # 0.15 <= omega_m < 0.45
    # -0.8 <= w0 <= -1.2
    # 0.5 <= sigma_8 <= 1.0
    generate_cfg_file(args.output_file,\
            args.omega_m, \
            args.w0, \
            args.sigma8, \
            args.rand_seed)
