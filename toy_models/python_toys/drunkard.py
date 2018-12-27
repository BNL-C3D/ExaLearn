
"""
Author: Yihui (Ray) Ren
Email : yihui.ren [at] gmail.com
"""

import math
import argparse
import numpy as np

def step(v, w, dim_sz, delta, batch:int) -> int:
    #max_dist = 0
    instructions = np.random.choice(2*dim_sz, batch)
    for x in instructions:
        i = x//2
        s = delta if x%2==0 else -delta
        w[i] += 2*s*v[i] + s*s
        v[i] += s
        #max_dist = max( max_dist, w.sum() )
        #max_dist = max( max_dist, w.sum() )
    #return max_dist

def drunkard(n:int, D:int=2, seed:int=None, batch:int=1<<20) -> float:
    np.random.seed(seed)
    v = np.zeros(D, dtype=np.float64);
    w = np.zeros(D, dtype=np.float64);
    ans = 0.0

    #ans = max( ans, step(v, w, D, 1.0, n%batch) )
    step(v, w, D, 1.0, n%batch) 
    for _ in range(n//batch):
        #ans = max( ans, step(v, w, D, 1.0, batch) )
        step(v, w, D, 1.0, batch) 
    #return np.sqrt(ans)
    ans = np.sqrt(w.sum())
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='drunkard.py', usage='%(prog)s [options]')
    parser.add_argument('-n', type=int, required=True, help="number of steps" )
    parser.add_argument('-m', type=int, required=True, help="number of trials")
    parser.add_argument('-d', '--dimension', required=False, \
            type=int, default=2,  help="number of dimensions")
    parser.add_argument('-s', '--seed', required=False, \
            type=int, default=None,  help="random seed")
    parser.add_argument('-b', '--batch_size', required=False,\
            type=int, default=1<<20, help="batch size")
    args = parser.parse_args()

    tot = 0.0
    for _ in range(args.m):
        ans = drunkard(args.n, args.dimension, args.seed, args.batch_size)
        tot += ans
    print( tot/args.m )
    d = args.dimension
    print( math.sqrt(2*args.n / d) * math.exp( math.lgamma((d+1)/2) - math.lgamma(d/2)) )
    ## E[max_deviation] = tot/args.m
    ## E[] / sqrt(n) = sqrt(pi/2)
    ## E[final_distance] = sqrt(2n/d)*gamma((d+1)/2) / gamma(d/2)

