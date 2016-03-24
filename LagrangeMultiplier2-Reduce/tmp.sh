#! /bin/sh
exe=$1
npack=$2
#npack=`echo "$npack/$np" | bc`
ne1=$3
ne2=0

ulimit -s unlimited

p=$4
c=$5
t=$6
filter=$7
ca=$c
ta=4
domain=$((${ca}*${ta}))
dc=$domain

echo -e "p=$p\nc=$c\nt=$t\ndomain=$domain\n"

unset I_MPI_PIN_DOMAIN
unset KMP_AFFINITY
unset KMP_PLACE_THREADS
unset OMP_NESTED
unset MKL_DYNAMIC
unset OMP_DYNAMIC
unset MKL_NUM_THREADS
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false
export OMP_NESTED=true
#export OMP_AFFINITY=spread
#export OMP_PLACES=threads
#export OMP_PLACES=cores($c)


#export OMP_PLACES={1:$t}:$c:$ta
export OMP_PLACES="cores($c)"
export OMP_PROC_BIND=spread,close
#export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=${c},${t}

#export OLD_KMP_PLACE_THREADS=${ca}Cx${ta}T,0O,
####export KMP_PLACE_THREADS=1s@0,${ca}c@0,${ta}t
#export KMP_PLACE_THREADS=1s@0,${ca}c@0,${ta}t
#export KMP_PLACE_THREADS=${ca}c@0,${ta}

#Ns[@N],Nc[@N],Nt
#export KMP_AFFINITY=compact,granularity=fine,
#export KMP_AFFINITY=balanced,granularity=fine,
#export KMP_AFFINITY=scatter,granularity=fine,
#export KMP_AFFINITY=compact,
#export KMP_AFFINITY=scatter,
#export KMP_AFFINITY=balanced,
export KMP_AFFINITY=${KMP_AFFINITY}verbose
#this is causing the warning about KMP_PLACE_THREADS

export I_MPI_PIN_DOMAIN=${domain}:compact


#echo "KMP_AFFINITY=$KMP_AFFINITY"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
#echo "OLD KMP_PLACE_THREADS=$OLD_KMP_PLACE_THREADS"
echo "KMP_PLACE_THREADS=$KMP_PLACE_THREADS"
echo "I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN"

echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; 
if [[ -z "$filter" ]]
then
mpiexec.hydra -n $p $exe < tmpparams 
else
mpiexec.hydra -n $p $exe < tmpparams | grep -v "ALIVE" | grep -v "Reset" | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
fi

