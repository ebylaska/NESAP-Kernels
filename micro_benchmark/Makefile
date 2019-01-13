SUFFIX=.mic

CXX = mpiicpc

LIBRARY = -mkl

CXXOPTIONS =  -O3 -openmp 
CXXOPTIONS += -mmic 
#call MKL or FMF matrix product. 
#FMF does three matrix products so 3x the flops of MKL
CXXOPTIONS += -DCALL_MKL

all: dgemm_simple

dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

clean:
	rm dgemm_simple.x${SUFFIX} 


