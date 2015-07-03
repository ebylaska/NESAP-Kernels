SRCDIR = .
OBJECTS = main.o D3dB.o hilbert.o hcurve.o olist.o Parallel-mpi.o 

LIBDIR1 = ../lib
LIBRARY = -lfftpack

OPTIONS = -O3
FFLAGS =  $(OPTIONS) -L$(LIBDIR1) 
FC = mpif77

main.x:   ${OBJECTS} 
	$(FC) -o main-mpi.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean: 
	rm -f core phase-mpi.x ${OBJECTS} $(TOBJECTS)


