SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o

LIBRARY = -lblas


OPTIONS = -O3  
FFLAGS =  -fopenmp -DUSE_OPENMP $(OPTIONS) 
#FFLAGS =  $(OPTIONS) 
FC = mpif90

lmbda_test.x: ${OBJECTS}
	$(FC) -o lmbda_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean:
	rm ${OBJECTS} lmbda_test.x


