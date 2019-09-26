#!/bin/sh
#SBATCH -N 1
#SBATCH -J pycola
#SBATCH -t 1-2:00:00 # 1day 2 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user <yourname>@<your.domain>

module load anaconda3
source activate pycola

#omega=(0.15 0.20 0.25 0.30 0.35 0.40)
omega=(0.150 0.175 0.200 0.225 0.250 0.275 0.300 0.325 0.350 0.375 0.400)
sigma=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90)
#wwwww=(-1.2 -1.1 -1.0 -0.9 -0.8)

for m in ${omega[@]};
do
    for sig in ${sigma[@]};
    do
        echo $m $sig
        python run.py -m $m -w -0.1 -s $sig
    done
done

