#! /bin/sh
#exe is the binary you want to run
exe=$1

ulimit -s unlimited

processes=(1) # 4 2)
  minthreads=2
  maxthreads=2
  maxmaxth=16

  minth=2
  maxth=2
#16
#$maxthreads

  if [[ "$#" -ge 4 ]]; then
  minthreads=$4
  if [[ "$#" -ge 5 ]]; then
  maxthreads=$5
  if [[ "$#" -ge 6 ]]; then
  mincores=$6
  if [[ "$#" -ge 7 ]]; then
  maxcores=$7
  if [[ "$#" -ge 8 ]]; then
  minth=$8
  if [[ "$#" -ge 9 ]]; then
  maxth=${9}
  fi
  fi
  fi
  fi
  fi
  fi



#for ip in `seq $minthreads $maxthreads`
  for ip in ${processes[@]} #`seq $minprocess $maxprocess`
  do
  mincores=$minthreads
  maxcores=$maxthreads
#$(($maxmaxth/$ip))
#if [[ "$ip" -ne "$minthreads" ]] ; then
#  mincores=1
#fi

#  if [[ "$#" -lt 7 ]]; then
#    maxcores=$(($maxthreads/$ip))
#  fi

  npack=$onpack

  for ic in `seq $mincores $maxcores`
  do
  for it in `seq $minth $maxth`
  do
prod=$((${ip}*${ic}*${it}))
taomp=$(($it>4?4:$it))
  ca=$(printf "%.0f\n" $(echo "scale=2;($ic*$it)/$taomp +0.49" | bc))
proda=$((${ip}*${ca}*${taomp}))
#echo $ip $proda vs $maxmaxth
  if [ "$maxmaxth" -ge "$proda"  ]
  then
  p=$ip
  t=$it
  c=$ic
#ca=$c
  ta=$t


#$it
  ta=$(($maxmaxth/$prod))
ta=$(($ta>4?$ta:4))
#ta=$(($t>4?$t:4))


domain=$((${ca}*${ta}))
domain=$((${ca}*${taomp}))
  dc=$domain


  unset OMP_NESTED
  unset MKL_DYNAMIC
  unset KMP_PLACE_THREADS
  unset OMP_DYNAMIC
  unset MKL_NUM_THREADS
  export MKL_DYNAMIC=false
  export OMP_DYNAMIC=false
  export OMP_NESTED=true

  export OMP_NUM_THREADS=${c},${t}




###        export OMP_PLACES="{0:$taomp}:$ca:$taomp"
#        export OMP_PLACES="cores($ca)"
        export OMP_PLACES="cores"
#        export OMP_PLACES="threads"
#echo $ip $domain $OMP_NUM_THREADS $OMP_PLACES
#echo $ip $domain $OMP_NUM_THREADS $OMP_PLACES 1>&2 
#
        export OMP_PROC_BIND=spread,close
#        export OMP_PROC_BIND=close
##       export I_MPI_PIN_DOMAIN=auto
##       export I_MPI_PIN_DOMAIN=omp
#

       export I_MPI_PIN_DOMAIN=${domain}:compact

#
##       export I_MPI_DEBUG=4

  export KMP_AFFINITY=${KMP_AFFINITY},verbose 

  mpiexec.hydra -n $p $exe  1> out.$(hostname).tmp 2> err.$(hostname).tmp

  fi
  done
  done
  done
