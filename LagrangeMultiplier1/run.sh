#! /bin/sh
exe=$1
np=$2
nc=$3
npack=$4
ne1=$5
ne2=$6
#export OMP_NUM_THREADS=$nc; echo -n "$np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun -np $np $exe < tmpparams | tail -n 3 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
#intel mic
#export OMP_NUM_THREADS=$nc; export KMP_AFFINITY=balanced; echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun.mic -n $np -ppn $np -hosts `hostname`-mic0 $exe < tmpparams 
#| tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
export OMP_NUM_THREADS=$nc; echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun -n $np $exe < tmpparams 
