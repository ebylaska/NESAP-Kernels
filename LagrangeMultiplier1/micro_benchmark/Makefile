SUFFIX=.reduce

CXX = mpiicpc

LIBRARY = -mkl
#LIBRARY =  -fopenmp -L${HOME}/software/OpenBLAS -lopenblas -fopenmp
BLISLIBRARY = -L${PREFIX}/blis/build/lib -lblis

CXXOPTIONS =  -O3 -openmp -g 
#CXXOPTIONS =  -O3 -openmp -g 
CXXOPTIONS += -mmic 
#call MKL or FMF matrix product. 
CXXOPTIONS += -DCALL_BLAS
CXXOPTIONS += -DNESTED
#CXXOPTIONS += -DMY_NESTED
#CXXOPTIONS += -DMKL

CXXOPTIONS += -DOMP_TASK
#CXXOPTIONS += -DNO_BLAS
CXXOPTIONS += -I. -fpermissive

CXXOPTIONS += -DAdd_
BLISCXXOPTIONS += -DBLIS
BLISCXXOPTIONS += -I${PREFIX}/blis/build/include/blis

all: dgemm_simple

dgemm_simple: dgemm_simple.cpp
	$(CXX) -o dgemm_simple.x${SUFFIX} ${CXXOPTIONS} dgemm_simple.cpp ${LIBRARY}

dgemm_simple_nomkl: dgemm_simple_nomkl.cpp
	$(CXX) -o dgemm_simple_nomkl.x${SUFFIX} ${CXXOPTIONS} dgemm_simple_nomkl.cpp ${LIBRARY}

dgemm_simple_blis: dgemm_simple_blis.cpp
	$(CXX) -o dgemm_simple_blis.x${SUFFIX} ${BLISCXXOPTIONS} ${CXXOPTIONS} dgemm_simple_blis.cpp ${BLISLIBRARY}

test_gemm: test_gemm.c
	$(CXX) -o test_gemm.x${SUFFIX} ${BLISCXXOPTIONS} ${CXXOPTIONS} test_gemm.c ${BLISLIBRARY}
clean:
	rm dgemm_simple.x${SUFFIX} 


