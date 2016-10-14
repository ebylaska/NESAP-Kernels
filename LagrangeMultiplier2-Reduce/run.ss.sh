#! /bin/sh
#exe is the binary you want to run
exe=$1
onpack=$2
ne1=$3
ne2=0

#echo $!

ulimit -s unlimited

if [ "$#" -lt 3 ]; then
  echo "Illegal number of parameters"
  exit
fi

processes=(1) # 4 2)
minthreads=68
maxthreads=68
maxmaxth=272

minth=1
maxth=1

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
  npack=$onpack

  for ic in `seq $mincores $maxcores`
  do
    for it in `seq $minth $maxth`
    do
      prod=$((${ip}*${ic}*${it}))

      taomp=$(($it>4?4:$it))
      ca=$(printf "%.0f\n" $(echo "scale=2;($ic*$it)/$taomp +0.49" | bc))
      proda=$((${ip}*${ca}*${taomp}))

      proda=$prod
      if [ "$maxmaxth" -ge "$proda"  ]
      then
        p=$ip
        t=$it
        c=$ic
        #ca=$c
        ta=$t


#$it
#        ta=$(($maxmaxth/$prod))
        ta=$(($ta>4?$ta:4))
        #ta=$(($t>4?$t:4))


        domain=$((${ca}*${ta}))
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
#        export OMP_PLACES="threads"
#echo $ip $domain $OMP_NUM_THREADS $OMP_PLACES
#echo $ip $domain $OMP_NUM_THREADS $OMP_PLACES 1>&2 
#



       export OMP_PLACES="cores"
       export OMP_PROC_BIND=spread,close
       export I_MPI_PIN_DOMAIN=auto
      
#export I_MPI_PIN_MODE=pm                # process launcher sets pinning by system-specific means (default) 
#export I_MPI_PIN_DOMAIN=auto:compact    # see https://software.intel.com/en-us/articles/mpi-and-process-pinning-on-xeon-phi
#export OMP_PLACES=threads               # treat each hyperthread as a place an OpenMP task can go
#export OMP_PROC_BIND=spread             # spread threads as widely as possible (avoid sharing cores,cache etc)
#export KMP_AFFINITY=noverbose,warnings,respect,granularity=fine,duplicates,scatter,0,0
#                                          # see https://software.intel.com/en-us/node/522691 

        echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; 

        #echo -n "$npack ${ne1} ${ne1} $p $c $t    "; 
        #mpirun -np $p $exe < tmpparams 2>&1 | tee out.$(hostname).tmp  | grep -v "cluster" | grep -v "ALIVE" | grep -v "Reset" | tail -n 8 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'



        #mpirun -np $p sde -knl -d -iform 1 -omix my_mix.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- $exe < tmpparams 
        mpirun -np $p amplxe-cl -start-paused -r my_vtune -collect memory-access -no-auto-finalize -trace-mpi --  $exe < tmpparams
#        amplxe-cl -collect advanced-hotspots -- $exe < tmpparams


# 2>&1 | tee out.$(hostname).tmp  | grep -v "cluster" | grep -v "ALIVE" | grep -v "Reset" | tail -n 8 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
        #mpiexec.hydra -n $p $exe < tmpparams 2>&1 | tee out.$(hostname).tmp  | grep -v "cluster" | grep -v "ALIVE" | grep -v "Reset" | tail -n 6 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
        #cat out.$(hostname).tmp  | grep -v "ALIVE" | grep -v "Reset" | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
	#exit
#        export OMP_PLACES="{0:$ta}:$ca:$ta"
#        echo -n "$npack ${ne1} ${ne1} $p $c $t    "; 
#        mpiexec.hydra -n $p $exe < tmpparams 2>&1 | tee out.$(hostname).tmp  | grep -v "cluster" | grep -v "ALIVE" | grep -v "Reset" | tail -n 6 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
	
        #export OMP_PLACES="{0:$t}:$ca:$t"
        #echo -n "$npack ${ne1} ${ne1} $p $c $t    "; 
        #mpiexec.hydra -n $p $exe < tmpparams 2>&1 | tee out.$(hostname).tmp  | grep -v "cluster" | grep -v "ALIVE" | grep -v "Reset" | tail -n 6 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
      fi
    done
  done
done
