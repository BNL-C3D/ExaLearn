#include <vector>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <unordered_map>
#include <random>
#include <bitset>
#include <limits>
#include <iostream> 
#include <iomanip>  
#include <string>
#include <sstream>  
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstring>
#include <cmath>
#include <cassert>

/* MPI */
#ifdef USE_MPI
#include <mpi.h>
#endif
/*******/


#define INF                         (int)1000000007
#define EPS                         1e-9

#define bg     begin()
#define pb     push_back
#define mp     make_pair

#define all(c)                      c.begin(), c.end()
#define forall(i,a,b)               for(int i=a;i<b;i++)
#define in(a,b)                     ( (b).find(a) != (b).end())
#define input( a )                  for( auto & x : a ) cin >> x;

using namespace std;

typedef vector<int> vi;
typedef pair<int,int> ii;
typedef vector<vi>  vvi;
typedef vector<ii>  vii;

#ifdef DEBUG
#define debug(args...)            {dbg,args; clog<<endl;}
#define print_( a )               for( auto & x : a ) clog << x << ' '; clog << '\n';
#define printPair_( a )           for( auto & x : a ) clog << '(' << x.first << ',' << x.second << ')' << ' '; clog << '\n';
#else
#define debug(args...)             // Just strip off all debug tokens
#define print_( a )               // skip
#define printPair_( a )           // skip
#endif
struct debugger
{
  template<typename T> debugger& operator , (const T& x)
  {    
    clog << x << " ";    
    return *this;    
  }
} dbg;

// std::ios::sync_with_stdio(false);
// std::cin.tie(NULL);
/******* Actual Code Starts Here *********/

struct UF {
	vi e;
	UF(int n) : e(n, -1) {}
	bool same_set(int a, int b) { return find(a) == find(b); }
	int size(int x) { return -e[find(x)]; }
	int find(int x) { return e[x] < 0 ? x : e[x] = find(e[x]); }
	void join(int a, int b) {
		a = find(a), b = find(b);
		if (a == b) return;
		if (e[a] > e[b]) swap(a, b);
		e[a] += e[b]; e[b] = a;
	}
};

random_device rd;
mt19937 rnd( rd() );
uniform_real_distribution<double> dist(0.0, 1.0);

double ergccsz(int n, double p){
  vector<double> v(n*(n+1)/2);
  for(auto & x : v ) x = dist(rnd);
  UF uf(n);
  int idx = 0, ans = 0;
  forall(i,0,n) forall(j,i+1,n) if( v[idx++] < p ) uf.join(i,j);
  forall(i,0,n) ans = max(ans, uf.size(i));
  return (double)(ans)/n;
}

double mean( vector<double> & v ){
  return accumulate(all(v),0.0)/v.size();
}

double stddev( vector<double> & v, double mean ){
  for(auto & x : v ) { x-=mean; x*=x; }
  double tot = accumulate(all(v),0.0);
  return sqrt(tot/(v.size()-1));
}

int main( int argc, char * argv[] ){
  
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  int rank, world_sz;
  MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  if( argc < 4 ){ cerr << "argc less than 4 ./main 100 10 0.1\n"; return 1; }
  int n = stoi(argv[1]), m = stoi(argv[2]);
  double p = stod(argv[3]); 

#ifdef USE_MPI
  int m_per_rank = m / world_sz;
  int data_cnt   = m_per_rank+1;
  vector<double> lcc_sz(data_cnt, -1.0);
  vector<double> sz( data_cnt*world_sz );
  if( rank < m%world_sz ) m_per_rank += 1;
  forall(i,0,m_per_rank){
    lcc_sz[i] = ergccsz(n,p);
  }

  MPI_Gather( &lcc_sz[0], data_cnt, MPI_DOUBLE, &sz[0], data_cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if( rank == 0 ){
    sz.erase( remove_if( all(sz), [](double x){ return x < 0; } ), sz.end() );
    double avg = mean(sz);
    double std = stddev(sz,avg);
    cout << avg << ' ' << std << '\n';
  }
  MPI_Finalize();
#else
  vector<double> sz;
  while(m-->0) sz.pb(ergccsz(n,p));
  double avg = mean(sz);
  double std = stddev(sz, avg);
  cout << avg << ' ' << std << '\n';
#endif
  return 0;
}
