#! /bin/sh
#exe is the binary you want to run
exe=$1

npack=$2
ne1=$3


ne2=0

ulimit -s unlimited

if [ "$#" -lt 3 ]; then
  echo "Illegal number of parameters"
  exit
fi

minthreads=1
maxthreads=240
mincores=1

minth=1
maxth=240
maxth=4
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



for ip in `seq $minthreads $maxthreads`
do

  if [[ "$ip" -ne "$minthreads" ]] ; then
    mincores=1
  fi

  if [[ "$#" -lt 7 ]]; then
    maxcores=$(($maxthreads/$ip))
  fi

  for ic in `seq $mincores $maxcores`
  do
    for it in `seq $minth $maxth`
    do
      prod=$((${ip}*${ic}*${it}))
      if [ "$maxthreads" -ge "$prod"  ]
      then
        p=$ip
        t=$it
        c=$ic
        ca=$c
        ta=$t

        ta=$(($maxthreads/$prod))
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
        export OMP_PLACES="cores"
        export OMP_PROC_BIND=spread,close
        export I_MPI_PIN_DOMAIN=${domain}:compact
        
        

        echo -n "$npack ${ne1} ${ne1} $p $c $t    "; 
        echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; 
        mpiexec.hydra -n $p $exe < tmpparams > out.tmp
        cat out.tmp | grep -v "ALIVE" | grep -v "Reset" | tail -n 5 | awk '{print $(NF-3) }' | tr '\n' "   " 
        cat out.tmp | grep -v "ALIVE" | grep -v "Reset" | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
        cp out.tmp out_${p}_${c}_${t}_${SLURM_JOB_ID}.dat
#        echo $KMP_PLACE_THREADS >> out_${p}_${c}_${t}_${SLURM_JOB_ID}.dat 
      fi
    done
  done
done
