SRCDIR = .
OBJECTS = main.o D3dB.o hilbert.o hcurve.o olist.o Parallel-mpi.o 

LIBDIR1 = /usr/local/lib
LIBRARY = -lfftw3

OPTIONS = -O3
FFLAGS =  $(OPTIONS) -L$(LIBDIR1) 
FC = mpif90

main.x:   ${OBJECTS} 
	$(FC) -o main-mpi.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean: 
	rm -f core phase-mpi.x ${OBJECTS} $(TOBJECTS)


