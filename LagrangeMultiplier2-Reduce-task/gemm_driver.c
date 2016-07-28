#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sched.h>
#include <mpi.h>

#include <omp.h>

int get_core(void) {
    int core;
    core = sched_getcpu();
    if (core < 0) {
        perror("affinity");
        exit(1);
    }
    return core;
}

int main(int argc, char ** argv) {
double *  A,*B,*C;
int m,n,k;
int mpirank;

MPI_Init(&argc,&argv);

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
#pragma omp parallel
    {
        int outer_tid;
        outer_tid = omp_get_thread_num();
#pragma omp parallel 
        {
            int inner_tid;
            int core;
            inner_tid = omp_get_thread_num();
            core = get_core();
            printf("mpi rank %d, outer thread %d, inner thread %d is running on core %d\n", mpirank,
                   outer_tid, inner_tid, core);
        }
    }


    A = (double*)malloc(m*k*sizeof(double));
    B = (double*)malloc(n*k*sizeof(double));
    C = (double*)malloc(m*n*sizeof(double));
#pragma omp parallel
    {
        int outer_tid;
        outer_tid = omp_get_thread_num();
#pragma omp parallel 
        {
            int inner_tid;
            inner_tid = omp_get_thread_num();
            double one = 1.0;
            dgemm_omp_("N","N", &m,&n,&k,&one,A,&m,B,&k,&one,C,&n);
        }
    }


    free(C);
    free(B);
    free(A);


MPI_Finalize();
}
