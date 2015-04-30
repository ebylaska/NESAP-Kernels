SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o 

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


CXXOPTIONS =  -O3 -openmp -DINTEL
OPTIONS =  -O3

#SUFFIX=.host

SUFFIX=.mic
CXXOPTIONS += -mmic 
OPTIONS += -mmic

#-DMKL_DIRECT_CALL_SEQ 
#OPTIONS = -O0 -g3 
#OPTIONS += -DBIG_GEMM
#OPTIONS += -DOMP_TASK
CXXOPTIONS += -DOMP_TASK
#OPTIONS += -heap-arrays 1024
OPTIONS += -align array64byte
FFLAGS =  -openmp -DUSE_OPENMP -DINTEL $(OPTIONS) 
#FFLAGS =  -fopenmp -DUSE_OPENMP $(OPTIONS) 
#FFLAGS =  $(OPTIONS) 
FC = mpiifort
CXX = mpiicpc
#FC = mpif90.mpich


lmbda_test.x: ${OBJECTS}
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} ${OBJECTS} ${LIBRARY}


dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}


#	$(FC) -o dgemm_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}

clean:
	rm ${OBJECTS} 


