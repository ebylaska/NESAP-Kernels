SRCDIR = .
OBJECTS = lmbda_test.o Parallel-mpi.o nwpw_timing.o pack.o psi_lmbda.o 

#LIBRARY = ${HOME}/software/OpenBLAS/libopenblas.a
#LIBRARY = -lblas
LIBRARY = -mkl


CXXOPTIONS =  -O3 -g -fopenmp -DINTEL
OPTIONS =  -O3 -g

#SUFFIX=.host
SUFFIX=.mic
CXXOPTIONS += -xMIC_AVX512 
OPTIONS += -xMIC_AVX512

#-DMKL_DIRECT_CALL_SEQ 
OPTIONS += -O3 -g3 
#OPTIONS += -DBIG_GEMM
#OPTIONS += -DOMP_TASK
#CXXOPTIONS += -DOMP_TASK
#OPTIONS += -heap-arrays 1024
#OPTIONS += -align array64byte
FFLAGS =  -fopenmp -DUSE_OPENMP -DINTEL $(OPTIONS) -mkl 
#FFLAGS =  -fopenmp -DUSE_OPENMP $(OPTIONS) 
#FFLAGS =  $(OPTIONS) 
FC = mpiifort
CXX = mpiicpc
CC= mpiicc
#FC = mpif90.mpich

#LIBRARY += -L/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak -lmap-sampler-pmpi -lmap-sampler -Wl,--eh-frame-hdr
#FFLAGS += -Wl,-rpath=/chos/global/u2/m/mjac/NESAP-Kernels_babbage2/NESAP-Kernels/LagrangeMultiplier1.bak

OPTIONS+= -I${VTUNE_AMPLIFIER_XE_2016_DIR}/include
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








lmbda_test.x: $(ITTCOBJ) $(ITTF90OBJ) ${OBJECTS} 
	$(FC) -o lmbda_test.x${SUFFIX} ${FFLAGS} $(ITTCOBJ) $(ITTF90OBJ) ${OBJECTS} ${LIBRARY}

dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

api_fortran_itt: api_fortran_itt.o
	$(CC) -o api_fortran_itt ${CFLAGS} api_fortran_itt.o

#	$(FC) -o dgemm_test.x ${FFLAGS} ${OBJECTS} ${LIBRARY}

clean:
	rm -f affinity.o
	rm ${OBJECTS} 


