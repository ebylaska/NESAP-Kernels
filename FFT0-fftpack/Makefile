SRCDIR = .
OBJECTS = main.o D3dB.o hilbert.o hcurve.o olist.o Parallel-mpi.o util_date.o nwpw_timing.o  #blas.o

LIBDIR1 = ../lib
LIBRARY = -lfftpack -lblas

#OPTIONS = -O3 -ffast-math -Wuninitialized
#OPTIONS = -g 
OPTIONS = -O0 -g -fopenmp
#OPTIONS += -fno-automatic -fbounds-check 
#OPTIONS += -fmax-stack-var-size=327680
#OPTIONS += -frecursive

FFLAGS =  $(OPTIONS) -L$(LIBDIR1) 
FC = mpif90

main.x:   ${OBJECTS} 
	$(FC) -o main-mpi.x ${FFLAGS} ${OBJECTS} ${LIBRARY}


clean: 
	rm -f core main-mpi.x ${OBJECTS} $(TOBJECTS)


