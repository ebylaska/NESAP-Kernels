SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o 
#blas.o
#gemm.o


#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl=sequential -fopenmp  

CXXOPTIONS =  -O3 -g -fopenmp -DINTEL
OPTIONS =  -O3 -g -mkl=sequential

SUFFIX=.mic
#CXXOPTIONS += -xMIC_AVX512 
OPTIONS +=  -xMIC_AVX512
FFLAGS =  -fopenmp -DUSE_OPENMP -DINTEL $(OPTIONS)
FC = mpiifort
OPTIONS+= -I${VTUNE_AMPLIFIER_XE_2016_DIR}/include
CXX = mpiicpc
CC= mpiicc
CFLAGS=-fopenmp $(OPTIONS) 
CFLAGS += -xMIC_AVX512


FFLAGS+= -g -I.
CFLAGS+= -g -I$(VTUNE_AMPLIFIER_XE_2016_DIR)/include
LIBRARY+=$(VTUNE_AMPLIFIER_XE_2016_DIR)/lib64/libittnotify.a
ITTCOBJ= api_itt_sde.o
ITTF90OBJ= module_itt_sde.o



%.o: %.c
	${CC} -c ${CFLAGS} ${CCDEFS} $< 
%.o: %.cpp
	${CXX} -c $< ${CXXFLAGS} ${CPPDEFS} 
%.o: %.f
	${FC} -c ${FFLAGS} ${FDEFS} $<
%.o: %.F
	${FC} -c ${FFLAGS} ${FDEFS} $<
%.o: %.f90
	${FC} -c ${FFLAGS} ${FDEFS} $<







all: affinity lmbda_test.x

gemm_driver: gemm_driver.o gemm.o
	$(CC) -o gemm_driver ${CFLAGS} gemm_driver.o gemm.o ${LIBRARY}

lmbda_test.x: $(ITTCOBJ) $(ITTF90OBJ) ${OBJECTS} 
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} $(ITTCOBJ) $(ITTF90OBJ) ${OBJECTS} ${LIBRARY}

dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

affinity: affinity.o
	$(CC) -o affinity ${CFLAGS} affinity.o

api_fortran_itt: api_fortran_itt.o
	$(CC) -o api_fortran_itt ${CFLAGS} api_fortran_itt.o

clean:
	rm -f ${OBJECTS} 
	rm -f affinity.o
	rm -f api_fortran_itt.o

realclean: clean
	rm -f affinity
	rm -f api_fortran_itt
	rm -f lmbda_test.x${SUFFIX}
	rm -f dgemm_simple.x${SUFFIX}
