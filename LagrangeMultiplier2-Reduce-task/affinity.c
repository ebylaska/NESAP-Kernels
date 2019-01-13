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
MPI_Init(&argc,&argv);
int mpirank;
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
MPI_Finalize();
}
