# ER Random Graph model
## Intro
In the [Erdos-Renyi model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) a graph G(n, p) is constructed by connecting nodes randomly. Each edge is included in the graph with probability p independent from every other edge.
The threshold for connectedness is ln(n)/n, which means that a graph in G(n, p) will almost surely contain isolated vertices if p < ln(n)/n and thus be disconnected. Whereas, if p > ln(n)/n then a graph in G(n, p) will almost surely be connected.

## Compile
* `make`
* `make mpi`

## Run
* for example: `./er 100 1000 0.01`, the order of graph $n=100$, the MC runs $m=1000$, the percolation probability $p=0.01$.
* `mpirun -n 32 ./er 100 1000 0.01`, to run in parallel.
* the output is the average fraction of largest connected component, and its standard deviation.


