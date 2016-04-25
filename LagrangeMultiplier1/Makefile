SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o 

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


CXXOPTIONS =  -O3 -g -qopenmp -DINTEL
OPTIONS =  -O3 -g

#SUFFIX=.host
SUFFIX=.mic
CXXOPTIONS += -mmic 
OPTIONS += -mmic

#-DMKL_DIRECT_CALL_SEQ 
OPTIONS += -O3 -g3 
#OPTIONS += -DBIG_GEMM
#OPTIONS += -DOMP_TASK
#CXXOPTIONS += -DOMP_TASK
#OPTIONS += -heap-arrays 1024
#OPTIONS += -align array64byte
FFLAGS =  -qopenmp -DUSE_OPENMP -DINTEL $(OPTIONS) -mkl 
FC = mpiifort
CXX = mpiicpc
#FC = mpif90.mpich

#LIBRARY += -L/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak -lmap-sampler-pmpi -lmap-sampler -Wl,--eh-frame-hdr
#FFLAGS += -Wl,-rpath=/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak


lmbda_test.x: ${OBJECTS}
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} ${OBJECTS} ${LIBRARY}


dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}


clean:
	rm ${OBJECTS} 


realclean: clean
	rm -f lmbda_test.x${SUFFIX}
