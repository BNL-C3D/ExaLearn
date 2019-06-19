#!/bin/sh
#SBATCH -N 1
#SBATCH -J pycola
#SBATCH -t 1-2:00:00 # 1day 2 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user yren@bnl.gov

module load anaconda3
source activate pycola

omega=(0.15 0.20 0.25 0.30 0.35 0.40)
wwwww=(-1.2 -1.1 -1.0 -0.9 -0.8)

for m in ${omega[@]};
do
    for w in ${wwwww[@]};
    do
        echo $o $w
        python run.py -m $m -w $w -s 0.7
    done
done

