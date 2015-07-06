SUFFIX=.nomkl.mic

CXX = mpiicpc

LIBRARY = -mkl
BLISLIBRARY = -L${PREFIX}/blis/build/lib -lblis

CXXOPTIONS =  -O3 -openmp -g 
CXXOPTIONS += -mmic 
#call MKL or FMF matrix product. 
#FMF does three matrix products so 3x the flops of MKL
#CXXOPTIONS += -DCALL_MKL
CXXOPTIONS += -DOMP_TASK
CXXOPTIONS += -DNO_MKL
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


