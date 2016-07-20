SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o gemm.o

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl -fopenmp

CXXOPTIONS =  -O3 -g -fopenmp -DINTEL
OPTIONS =  -O0 -g -mkl

SUFFIX=.mic
#CXXOPTIONS += -xMIC_AVX512 
#OPTIONS +=  -xMIC_AVX512
FFLAGS =  -fopenmp -DUSE_OPENMP -DINTEL $(OPTIONS)
FC = mpiifort
CXX = mpiicpc
CC=mpiicc
CFLAGS=-fopenmp $(OPTIONS) 
#CFLAGS += -xMIC_AVX512





%.o: %.c
	${CC} -c ${CFLAGS} ${CCDEFS} $< 
%.o: %.cpp
	${CXX} -c $< ${CXXFLAGS} ${CPPDEFS} 
%.o: %.f
	${FC} -c ${FFLAGS} ${FDEFS} $<
%.o: %.F
	${FC} -c ${FFLAGS} ${FDEFS} $<







all: affinity lmbda_test.x

gemm_driver: gemm_driver.o gemm.o
	$(CC) -o gemm_driver ${CFLAGS} gemm_driver.o gemm.o ${LIBRARY}

lmbda_test.x: ${OBJECTS} 
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} ${OBJECTS} ${LIBRARY}


dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

affinity: affinity.o
	$(CC) -o affinity ${CFLAGS} affinity.o

clean:
	rm -f ${OBJECTS} 
	rm -f affinity.o

realclean: clean
	rm -f affinity
	rm -f lmbda_test.x${SUFFIX}
	rm -f dgemm_simple.x${SUFFIX}
