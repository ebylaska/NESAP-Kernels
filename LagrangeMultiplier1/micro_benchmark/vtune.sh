c=60; export OMP_NUM_THREADS=$c;export KMP_AFFINITY=balanced,granularity=fine; amplxe-cl -r OUT -collect advanced-hotspots -target-system=mic-host-launch -- ./dgemm_simple.x.mkl.mic < smallparams.dgemm 

