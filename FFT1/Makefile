SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


OPTIONS = -O3  
#OPTIONS = -O0 -g3 
#OPTIONS += -DBIG_GEMM
FFLAGS =  -openmp -DUSE_OPENMP -DINTEL $(OPTIONS) 
#FFLAGS =  -fopenmp -DUSE_OPENMP $(OPTIONS) 
#FFLAGS =  $(OPTIONS) 
FC = mpiifort
#FC = mpif90.mpich


lmbda_test.x: ${OBJECTS}
	$(FC) -o lmbda_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean:
	rm ${OBJECTS} lmbda_test.x


