p=$1
c=1
team=$2
export OMP_NUM_THREADS=$team
##export I_MPI_PIN_DOMAIN=socket
#export I_MPI_PIN_DOMAIN=core
#export KMP_AFFINITY=scatter
##export I_MPI_DEBUG=5
##unset OMP_PROC_BIND
##unset OMP_PLACES
##export KMP_AFFINITY=scatter
##export OMP_NUM_THREADS=$team,$c
#export OMP_NESTED=true
#export OMP_PLACES=cores
#export OMP_PLACES=threads
#export OMP_PROC_BIND=close
#export KMP_AFFINITY=verbose
##export OMP_PROC_BIND=spread,close
#export MKL_DYNAMIC=false
#export KMP_HOT_TEAMS_MAX_LEVEL=2
#export KMP_HOT_TEAMS_MODE=1
#export KMP_BLOCKTIME=infinite
##export I_MPI_PIN_DOMAIN=core

mpirun -n $p ../LagrangeMultiplier0/lmbda_test.x < tmpparams 
#mpirun -n $p ./dgemm_simple.x.reduce < smallparams.dgemm 

