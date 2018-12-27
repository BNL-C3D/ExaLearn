# ER Random Graph model
## Intro
[Erdos-Renyi model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)

## Compile
* `make`
* `make mpi`

## Run
* for example: `./er 100 1000 0.01`, the order of graph $n=100$, the MC runs $m=1000$, the percolation probability $p=0.01$.
* `mpirun -n 32 ./er 100 1000 0.01`, to run in parallel.
* the output is the average fraction of largest connected component, and its standard deviation.


