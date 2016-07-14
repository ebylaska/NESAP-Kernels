SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


CXXOPTIONS =  -O0 -g -fopenmp -DINTEL
OPTIONS =  -O0 -g

SUFFIX=.mic
CXXOPTIONS += -xMIC_AVX512 
OPTIONS +=  -xMIC_AVX512
FFLAGS =  -fopenmp -DUSE_OPENMP -DINTEL $(OPTIONS) -mkl 
FC = mpiifort
CXX = mpiicpc
CC=mpiicc
CFLAGS=-fopenmp -g -O0 -mkl -xMIC_AVX512

all: affinity lmbda_test.x

gemm.o: gemm.c
	$(CC) -o gemm.o -c ${CFLAGS} gemm.c

lmbda_test.x: ${OBJECTS} gemm.o
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} ${OBJECTS} gemm.o ${LIBRARY}


dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

affinity: affinity.o
affinity.o: affinity.c

clean:
	rm -f ${OBJECTS} 
	rm -f affinity.o

realclean: clean
	rm -f affinity
	rm -f lmbda_test.x${SUFFIX}
	rm -f dgemm_simple.x${SUFFIX}
