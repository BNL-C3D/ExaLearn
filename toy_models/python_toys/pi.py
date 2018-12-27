"""
Author: Yihui (Ray) Ren
Email : yihui.ren [at] gmail.com
"""

import math
import argparse
import numpy as np


def mc_cnt( batch: int ) -> int:
    x = np.power(np.random.rand(batch),2)
    y = np.power(np.random.rand(batch),2)
    return (x+y<1).sum()

def mc_pi(n: int, batch: int = 1<<20) -> float:
    cnt = mc_cnt(n%batch)
    for _ in range(n//batch):
        cnt += mc_cnt(batch)
    return 4*cnt/n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pi.py', usage='%(prog)s [options]')
    parser.add_argument('-m', type=int,  help="number of trials", required=True)
    parser.add_argument('--s', type=int, default=None,  help="random seed")
    parser.add_argument('--b', type=int, default=1<<20, help="batch size")
    args = parser.parse_args()
    np.random.seed(args.s)
    ans  = mc_pi(args.m, args.b)
    print(ans, "error=", abs(ans - math.pi))
