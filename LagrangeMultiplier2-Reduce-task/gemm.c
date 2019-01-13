#include <stdio.h>      /* Standard Library of Input and Output */
#include <complex.h>    /* Standard Library of Complex Numbers */
#include <math.h>  
#include<omp.h>

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

double gemmOMPThreshold = 16^3;



void zgemm_omp_task(char transa, char transb, int m, int n, int k, double complex alpha, double complex* a, int lda, double complex* b, int ldb, double complex beta, double complex* c, int ldc, int depth) {
//double task_recursion_cutoff_level = (log(omp_get_num_threads())/log(2) + 3 ) ;
double task_recursion_cutoff_level = (log2(omp_get_num_threads()) + 3 ) ;
	if (depth>=task_recursion_cutoff_level || (double)m*n*k <= gemmOMPThreshold)
		zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	else {
		char opA = transa=='T'||transa=='t'||transa=='C'||transa=='c';
		char opB = transb=='T'||transb=='t'||transb=='C'||transb=='c';
		if (n >= max(m,k)) {
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			zgemm_omp_task(transa, transb, m, n/2, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1);                                                            // C0 = alpha * A*B0 + beta * C0
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			zgemm_omp_task(transa, transb, m, n-n/2, k, alpha, a, lda, opB ? b+n/2 : b+(n/2)*ldb, ldb, beta, c+(n/2)*ldc, ldc, depth+1);                        // C1 = alpha * A*B1 + beta * C1
#pragma omp taskwait
		} else if (m >= k) {
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			zgemm_omp_task(transa, transb, m/2, n, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1);                                                            // C0 = alpha * A0*B + beta * C0
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			zgemm_omp_task(transa, transb, m-m/2, n, k, alpha, opA ? a+(m/2)*lda : a+m/2, lda, b, ldb, beta, c+m/2, ldc, depth+1);                              // C1 = alpha * A1*B + beta * C1
#pragma omp taskwait
		} else {
			zgemm_omp_task(transa, transb, m, n, k/2, alpha, a, lda, b, ldb, beta, c, ldc, depth);                                                         // C  = alpha * A0*B0 + beta * C
			zgemm_omp_task(transa, transb, m, n, k-k/2, alpha, opA ? a+k/2 : a+(k/2)*lda, lda, opB ? b+(k/2)*ldb : b+k/2, ldb, (double complex)1., c, ldc, depth); // C += alpha * A1*B1
		}
	}
}

void dgemm_omp_task(char transa, char transb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc, int depth,double task_recursion_cutoff_level) {
//double task_recursion_cutoff_level = (log(omp_get_num_threads()) + 3 ) / log(2);

//#pragma omp critical
//printf("Task %d %d %d %lf %x %d %x %d %lf %x %d %d %lf %lf\n", m, n,  k, alpha,  a, lda,  b, ldb,  beta,  c, ldc, depth,task_recursion_cutoff_level,gemmOMPThreshold);
//#pragma omp critical
//printf("BEGIN Task %d vs %lf\n",depth,task_recursion_cutoff_level);

	if (depth>=task_recursion_cutoff_level || (double)m*n*k <= gemmOMPThreshold)
	{

#ifdef MKL
	   int dummy = mkl_set_num_threads_local(1)
#endif
	//printf("actual gemm call\n");
	//	printf("%d %d %d %lf %x %d %x %d %lf %x %d %d %lf %lf\n", m, n,  k, alpha,  a, lda,  b, ldb,  beta,  c, ldc, depth,task_recursion_cutoff_level,gemmOMPThreshold);
		dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
	//printf("after actual gemm call\n");
#ifdef MKL
        //restore the number of threads
        dummy = mkl_set_num_threads_local(0)
#endif
	}
	else {
		char opA = transa=='T'||transa=='t'||transa=='C'||transa=='c';
		char opB = transb=='T'||transb=='t'||transb=='C'||transb=='c';
		if (n >= max(m,k)) {
//#pragma omp taskgroup
//{
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			dgemm_omp_task(transa, transb, m, n/2, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1,task_recursion_cutoff_level);                                                            // C0 = alpha * A*B0 + beta * C0
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			dgemm_omp_task(transa, transb, m, n-n/2, k, alpha, a, lda, opB ? b+n/2 : b+(n/2)*ldb, ldb, beta, c+(n/2)*ldc, ldc, depth+1,task_recursion_cutoff_level);                        // C1 = alpha * A*B1 + beta * C1
//}
#pragma omp taskwait
		} 
    else if (m >= k) {
//#pragma omp taskgroup
//{
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			dgemm_omp_task(transa, transb, m/2, n, k, alpha, a, lda, b, ldb, beta, c, ldc, depth+1,task_recursion_cutoff_level);                                                            // C0 = alpha * A0*B + beta * C0
#pragma omp task final(depth >= task_recursion_cutoff_level-1) mergeable
			dgemm_omp_task(transa, transb, m-m/2, n, k, alpha, opA ? a+(m/2)*lda : a+m/2, lda, b, ldb, beta, c+m/2, ldc, depth+1,task_recursion_cutoff_level);                              // C1 = alpha * A1*B + beta * C1
//}
#pragma omp taskwait
		} 
    else {
			dgemm_omp_task(transa, transb, m, n, k/2, alpha, a, lda, b, ldb, beta, c, ldc, depth,task_recursion_cutoff_level);                                                         // C  = alpha * A0*B0 + beta * C
			dgemm_omp_task(transa, transb, m, n, k-k/2, alpha, opA ? a+k/2 : a+(k/2)*lda, lda, opB ? b+(k/2)*ldb : b+k/2, ldb, (double)1., c, ldc, depth,task_recursion_cutoff_level); // C += alpha * A1*B1
		}
	}

//#pragma omp critical
//printf("END Task %d\n",depth);

}


void dgemm_omp_(char * transa, char * transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc) {
//	int i,j;
//    printf("%d %d %d %lf %x %d %x %d %lf %x %d\n", *m, *n,  *k, *alpha,  a, *lda,  b, *ldb,  *beta,  c, *ldc);
//	for(i = 0; i<*m; i++){
//		for(j = 0; j<*n; j++){
//			double t = c[i+j*(*ldc)];
//			c[i+j*(*ldc)] = t;
//		}
//	}
//	for(i = 0; i<*m; i++){
//		for(j = 0; j<*k; j++){
//			double t = a[i+j*(*lda)];
//			a[i+j*(*lda)] = t;
//		}
//	}
//	for(i = 0; i<*k; i++){
//		for(j = 0; j<*n; j++){
//			double t = b[i+j*(*ldb)];
//			b[i+j*(*ldb)] = t;
//		}
//	}
//	dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); 
//printf("--------------------------------------------\n");
//#pragma omp parallel
//{
  double task_recursion_cutoff_level = (log2(omp_get_num_threads()) + 3 ) ;
//	#pragma omp single nowait
//  printf("         NTHREADS = %d    --  %lf\n",omp_get_num_threads(), task_recursion_cutoff_level);
//#pragma omp taskgroup
//{
//	#pragma omp single nowait
	dgemm_omp_task(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc, 0,task_recursion_cutoff_level); 
//}
//}
//printf("--------------------------------------------\n");
//exit(-1);
}
