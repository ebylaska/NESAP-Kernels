SRCDIR = .
OBJECTS = fft_test.o Parallel-mpi.o nwpw_timing.o pack.o

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


fft_test.x: ${OBJECTS}
	$(FC) -o fft_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean:
	rm ${OBJECTS} fft_test.x


