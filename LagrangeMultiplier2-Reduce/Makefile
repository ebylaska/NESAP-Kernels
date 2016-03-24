SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o omp_dgemm.o 

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


CXXOPTIONS =  -O3 -g -openmp -DINTEL
OPTIONS =  -O3 -g

#SUFFIX=.host
SUFFIX=.mic
CXXOPTIONS += -mmic 
OPTIONS += -mmic

#-DMKL_DIRECT_CALL_SEQ 
OPTIONS += -DBLK_FMF=300 
#OPTIONS += -O0 -g3 
#OPTIONS += -DBIG_GEMM
#OPTIONS += -DOMP_TASK
#CXXOPTIONS += -DOMP_TASK
#OPTIONS += -heap-arrays 1024
#OPTIONS += -align array64byte
FFLAGS =  -openmp -DUSE_OPENMP -DINTEL $(OPTIONS) -mkl 
#FFLAGS += -DMKL
#FFLAGS =  -fopenmp -DUSE_OPENMP $(OPTIONS) 
#FFLAGS =  $(OPTIONS) 
FC = mpiifort
CXX = mpiicpc
CC=icc
CFLAGS=-fopenmp -g -O2

affinity: affinity.o
affinity.o: affinity.c

#FC = mpif90.mpich

#FFLAGS+= -DOPENMP_REDUCE
#SUFFIX=.mic.romp



#LIBRARY += -L/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak -lmap-sampler-pmpi -lmap-sampler -Wl,--eh-frame-hdr
#FFLAGS += -Wl,-rpath=/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak


lmbda_test.x: ${OBJECTS}
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} ${OBJECTS} ${LIBRARY}


dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

mxm_openmp: mxm_openmp.F omp_dgemm.F
	$(FC) -o mxm_openmp.x${SUFFIX} ${FFLAGS} mxm_openmp.F omp_dgemm.F ${LIBRARY} -lm

#	$(FC) -o dgemm_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}

clean:
	rm ${OBJECTS} 


