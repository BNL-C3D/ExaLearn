"""
Author: Yihui (Ray) Ren
Email : yihui.ren [at] gmail.com
"""

import math
from collections import deque
import argparse
import numpy as np

def giant_connected_component(g):
    n = len(g)
    visited = np.zeros(n, dtype=np.int8)
    ans = 0
    for i in range(n):
        if not visited[i]:
            tmp = 0
            q = deque()
            q.append(i)
            visited[i] = 1
            while len(q) :
                u = q.popleft()
                tmp += 1
                for j in range(n):
                    if g[u][j] == 1 and visited[j] == 0:
                        visited[j] = 1
                        q.append(j)
        ans = max(ans, tmp)
    return ans;

def er_random_graph(n: int, p: float):
    s = np.random.rand( n*(n-1)//2 );
    g = np.zeros((n,n), dtype=np.int8)
    idx = 0;
    for i in range(n):
        for j in range(i+1,n):
            if s[idx] < p :
                g[i][j] = 1
                g[j][i] = 1
            idx += 1    

    return giant_connected_component(g)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='erdos_renyi.py', usage='%(prog)s [options]')
    parser.add_argument('-n', type=int,  help="number of verticies", required=True)
    parser.add_argument('-p', type=float,  help="probability of connecting verticies", required=True)
    parser.add_argument('-m', type=int,  help="number of trials", required=True)
    parser.add_argument('--s', type=int, default=None,  help="random seed")
    parser.add_argument('--b', type=int, default=1<<20, help="batch size")
    args = parser.parse_args()
    np.random.seed(args.s)
    ans = []
    for _ in range(args.m):
        tmp  =er_random_graph(args.n, args.p)
        print(tmp)
        ans.append(tmp)
    ans = np.asarray(ans)
    print(np.mean(ans), np.std(ans))
