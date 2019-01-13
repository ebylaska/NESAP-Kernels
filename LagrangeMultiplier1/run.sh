#! /bin/sh
exe=$1
npack=$2
#npack=`echo "$npack/$np" | bc`
ne1=$3
ne2=$4

ulimit -s unlimited

if [ "$#" -lt 4 ]; then
  echo "Illegal number of parameters"
  exit
fi

minthreads=1
maxthreads=240
mincores=1

minth=1
maxth=4
if [[ "$#" -ge 5 ]]; then
  minthreads=$5
  if [[ "$#" -ge 6 ]]; then
    maxthreads=$6
    if [[ "$#" -ge 7 ]]; then
      mincores=$7
      if [[ "$#" -ge 8 ]]; then
        maxcores=$8
        if [[ "$#" -ge 9 ]]; then
          minth=$9
          if [[ "$#" -ge 10 ]]; then
            maxth=${10}
          fi
      fi
      fi
    fi
  fi
fi



for ip in `seq $minthreads $maxthreads`
#for ip in `seq 1 1`
do

  if [[ "$ip" -ne "$minthreads" ]] ; then
    mincores=1
  fi

  if [[ "$#" -lt 8 ]]; then
    maxcores=$(($maxthreads/$ip))
  fi

  for ic in `seq $mincores $maxcores`
  do
  
    #KMP_PLACE_THREADS need a correct offset
    for it in `seq $minth $maxth`
    #for it in `seq 4 4`
    do
    
      prod=$((${ip}*${ic}*${it}))
      
      
      if [ "$maxthreads" -ge "$prod"  ]
      then
        p=$ip
        t=$it
        c=$ic
        ca=$c
        ta=$t
        domain=$((${ca}*${ta}))
        dc=$domain
        
        
        unset OMP_NESTED
        unset MKL_DYNAMIC
        unset OMP_DYNAMIC
        unset MKL_NUM_THREADS
        export MKL_DYNAMIC=false
        export OMP_DYNAMIC=false
        export OMP_NESTED=true

        export OMP_NUM_THREADS=${c},${t}
        export OLD_KMP_PLACE_THREADS=${ca}Cx${ta}T,0O,
        ####export KMP_PLACE_THREADS=1s@0,${ca}c@0,${ta}t
        #export KMP_PLACE_THREADS=1s@0,${ca}c@0,${ta}t
        #Ns[@N],Nc[@N],Nt
        export KMP_AFFINITY=compact,granularity=fine
        #export KMP_AFFINITY=balanced,granularity=fine
        
        #this is causing the warning about KMP_PLACE_THREADS
        #export I_MPI_PIN_DOMAIN=${domain}:compact
        
        
        #echo "KMP_AFFINITY=$KMP_AFFINITY"
        #echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
        #echo "OLD KMP_PLACE_THREADS=$OLD_KMP_PLACE_THREADS"
        #echo "KMP_PLACE_THREADS=$KMP_PLACE_THREADS"

        #echo "mpirun.mic -n $p -ppn $p -hosts `hostname`-mic0 $exe < tmpparams"
        echo -n "$npack ${ne1} ${ne1} $p $c $t    "; 
        echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; 
        mpiexec.hydra -n $p $exe < tmpparams > out.tmp
        #echo "mpiexec.hydra -n $p $exe < tmpparams"
        #cat out.tmp
        cat out.tmp | grep -v "ALIVE" | grep -v "Reset" | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
        cp out.tmp out_${p}_${c}_${t}_${SLURM_JOB_ID}.dat
        echo $KMP_PLACE_THREADS >> out_${p}_${c}_${t}_${SLURM_JOB_ID}.dat 
        #echo -n "$npack ${ne1} ${ne1} $p $c $t    ";
        #mpirun.mic -n $p -ppn $p -hosts `hostname`-mic0 ../LagrangeMultiplier2-Reduce/lmbda_test.x.mic < tmpparams

        #exit

      fi
    
    done
  done
done


#echo -n "$npack ${ne1} $p $c $t    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun.mic -n $p -ppn $p -hosts `hostname`-mic0 $exe < tmpparams


#export OMP_NUM_THREADS=$nc; echo -n "$np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun -np $np $exe < tmpparams | tail -n 3 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
#intel mic
#export OMP_NUM_THREADS=$nc; export KMP_AFFINITY=balanced,granularity=fine; echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun.mic -n $np -ppn $np -hosts `hostname`-mic0 $exe < tmpparams | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
#export OMP_NUM_THREADS=$nc; export KMP_AFFINITY=balanced,granularity=fine; echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun -n $np -ppn $np -hosts `hostname`-mic0 $exe < tmpparams | tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'

#export OMP_NUM_THREADS=$nc; export KMP_AFFINITY=balanced,granularity=fine; mpirun.mic -n $np -ppn $np -hosts `hostname`-mic0 $exe < tmpparams 

#echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun.mic -n $np -ppn $np -hosts `hostname`-mic0 $exe < tmpparams 
#| tail -n 5 | awk '{print $(NF-2) }' | tr '\n' "   " | awk '{print $0}'
#export OMP_NUM_THREADS=$nc; echo -n "$npack $np $nc    "; echo -e "${npack}\n${ne1}\n${ne2}\n" > tmpparams; mpirun -n $np $exe < tmpparams 
