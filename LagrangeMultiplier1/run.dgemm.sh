for c in 1 2 4 8 16 30 60 120 180 240; do echo "---------$c-------------\n" ; export OMP_NUM_THREADS=$c;export KMP_AFFINITY=balanced,granularity=fine; mpirun.mic -n 1 -hosts `hostname`-mic0 ./dgemm_simple.x.mic < params.dgemm ; done >> full.dgemm.exp.mic