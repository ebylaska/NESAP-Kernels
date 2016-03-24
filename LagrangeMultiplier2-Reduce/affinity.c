#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sched.h>

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

int main(void) {
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
            printf("outer thread %d, inner thread %d is running on core %d\n",
                   outer_tid, inner_tid, core);
        }
    }
}
