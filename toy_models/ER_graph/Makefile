CXX=g++
all: er.cpp
	$(CXX) -std=c++14 -O3 er.cpp -o er 

mpi: er.cpp
	mpicc -std=c++11 -DUSE_MPI -o er er.cpp -lm -lstdc++ ## for gcc 4.8 in most hpc

debug: er.cpp
	$(CXX) -std=c++14 -DDEBUG err.cpp -o er 

copy:
	cat er.cpp | pbcopy

cp:
	cat er.cpp | xclip -selection clipboard


