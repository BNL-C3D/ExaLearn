
"""
Author: Yihui (Ray) Ren
Email : yihui.ren [at] gmail.com

use with caution, might be slow.
TODO: UI
"""

import math
import numpy as np
"""
L.  Niemeyer,  L.  Pietronero,  and  H.J. Wiessmann,  Phys. Rev. Lett. 52, 1033 (1984)
https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.52.1033
"""

N = 201
eta = 0.8

BNDR_P = 1
SITE_P = 0
phi = np.zeros((N+2, N+2), dtype=np.float32)
phi[0,:]   = BNDR_P
phi[N+1,:] = BNDR_P
phi[:,0]   = BNDR_P
phi[:,N+1] = BNDR_P

print(phi)

mid = (N+2)//2
phi[mid][mid] = SITE_P

has_occupied = np.zeros((N+2, N+2), dtype=np.int8)
has_occupied[mid][mid] = 1

M = 1000

phi_prime = np.copy(phi)
for _ in range(2*(N+2)):
    if _%1000 == 0: print(_)
    for i in range(1,N+1):
        for j in range(1,N+1):
            if not has_occupied[i][j]:
                phi_prime[i][j] = 1/4 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1])
    phi = np.copy(phi_prime)

for _ in range(M):
    choices = []
    weights = []
    for i in range(1,N+1):
        for j in range(1,N+1):
            if not has_occupied[i][j]:
                phi_prime[i][j] = 1/4 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1])
                if any( [has_occupied[i-1][j], has_occupied[i+1][j],\
                        has_occupied[i][j-1], has_occupied[i][j+1]]):
                    choices.append( (i,j) )
                    weights.append( phi_prime[i][j] )
    weights = np.asarray(weights) 
    weights = np.power(weights, eta)
    weights = weights / sum(weights)
    #print(_,':',weights)
    #print(_,':',choices)
    chosen = np.random.choice( np.arange(len(choices), dtype=np.int32), 1, p=weights )[0]
    chosen = choices[chosen]
    has_occupied[chosen] = 1
    phi_prime[chosen]    = SITE_P
    phi = np.copy(phi_prime)
    #print("chosen", chosen)

np.savetxt('dla_N'+str(N)+'phi.csv', phi)
np.savetxt('dla_N'+str(N)+'site.csv', has_occupied)


