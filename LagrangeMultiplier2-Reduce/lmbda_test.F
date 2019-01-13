
      subroutine lmbda_test_r(npack1,ne)
#ifdef INTEL
      USE IFPORT
#endif
      implicit none

*     **** arguments ****
      integer npack1, ne(2)

*     **** variables ****
      integer MASTER,taskid,np,tid,nthr,mthr
      parameter (MASTER=0)
      logical oprint

*     **** psi variables ****
      !integer npack1_max,nemax
      !parameter (npack1_max=npack1,nemax=ne(1))

      integer nida,nidb
      complex*16 psi1(npack1*ne(1))
      complex*16 psi2(npack1*ne(1))
      complex*16 dpsi1(npack1*ne(1))
      real*8 lmbda(ne(1)*ne(1)*2),tmp(ne(1)*ne(1)*8)

      integer ispin,n1(2),n2(2),ms
      integer i,j,k,n,indxj,indxk,shift,mpierr,provided,ierr
      integer number_tests
      real*8 w,ww,dte

*     **** matrix_blocking common block ****
      integer mblock(2),nblock(2),algorithm(2)
      common /matrix_blocking/ mblock,nblock,algorithm

*     **** external functions ****
      integer  Parallel_maxthreads,Parallel_index_1dblock
      external Parallel_maxthreads,Parallel_index_1dblock
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      !**** testing parameters ****
      ispin = 1
      dte    = 1.0d-2 !* maximum component change in psi
      number_tests = 5
      !**** testing parameters ****




      n1(1) = 1
      n2(1) = ne(1)
      n1(2) = ne(1)+1
      n2(2) = ne(1)+ne(2)

      !call Parallel_Init()


      call Parallel_np(np)
      mthr = Parallel_maxthreads()

      call Parallel_taskid(taskid)
      oprint = (taskid.eq.MASTER)

      shift = ne(1)*npack1
      nida = 0
      if (taskid.eq.MASTER) nida = 1
      nidb = npack1-nida

*     **** define blocking for dgemm matrix multiply ****
      algorithm(1) = -1
      algorithm(2) = -1
      do ms=1,ispin
         if (ne(ms).gt.(mthr*np)) then
            algorithm(ms) = 1
            call Parallel_matrixblocking(mthr*np,ne(ms),ne(ms),
     >                                mblock(ms),nblock(ms))
         else if (ne(ms).gt.(mthr)) then
            algorithm(ms) = 0
            call Parallel_matrixblocking(mthr,ne(ms),ne(ms),
     >                                mblock(ms),nblock(ms))
         else
            algorithm(ms) = -1
         end if
c         write(*,*) ms," ne,ne --> mb,nb=",
c     >              ne(ms),ne(ms), "--> ",mblock(ms),nblock(ms)
c         write(*,*) "i=0:",Parallel_index_1dblock(ne(ms),mblock(ms),0)
c         write(*,*) "i=1:",Parallel_index_1dblock(ne(ms),mblock(ms),1)
c         write(*,*) "i=2:",Parallel_index_1dblock(ne(ms),mblock(ms),2)
      end do

c     +++++++++++++++++++++++++++++++++++
c     ++++ inititalize psi1 and psi2 ++++
c     +++++++++++++++++++++++++++++++++++
      do i=1,npack1*(ne(1)+ne(2))
         psi1(i)  = 2.0d0*rand() - 1.0d0
         dpsi1(i) = 2.0d0*rand() - 1.0d0
      end do

      !***  psi1 = Gram-Schmidt orthonormal ***
      do ms=1,ispin
         do k=ne(ms),1,-1
            indxk = 1 + (k-1)*npack1+(ms-1)*shift
            call epack_cc_dot(nida,nidb,psi1(indxk),psi1(indxk),w)
            w = 1.0d0/dsqrt(w)
            call dscal(2*npack1,w,psi1(indxk),1)
            do j=k-1,1,-1
               indxj = 1 + (j-1)*npack1+(ms-1)*shift
               call epack_cc_dot(nida,nidb,psi1(indxk),psi1(indxj),w)
               w = -w
               call daxpy(2*npack1,w,psi1(indxk),1,psi1(indxj),1)
            end do
         end do
      end do



!      write(*,*) "orthonormal passed"

      !*** dpsi1 = error ***
      ww = 0.0d0
      do k=1,(ne(1)+ne(2))
         indxk = 1 + (k-1)*npack1
         call epack_cc_dot(nida,nidb,dpsi1(indxk),dpsi1(indxk),w)
         if (dabs(w).gt.dabs(ww)) ww = w
      end do
      ww = 1.0d0/ww

      !write(*,*) "dot products", 2*npack1*(ne(1)+ne(2)), ww, dpsi1
      call dscal(2*npack1*(ne(1)+ne(2)),ww,dpsi1,1)
!      write(*,*) "dscal passed"
c     -----------------------------------
c     ---- inititalize psi1 and psi2 ----
c     -----------------------------------
      !if (oprint) write(*,*) "<psi1|psi1> = "
      !call check_norms(oprint,ispin,ne,nida,nidb,psi1)


      if (oprint) then
         write(*,*)
         write(*,*)
         write(*,*) "Lagrange mutiplier testing"
         write(*,*) "--------------------------"
         write(*,*) "number of cpus       =",np
         write(*,*) "max number of threads=",mthr
         write(*,*) "ispin                =",ispin
         write(*,*) "ne(1)                =",ne(1)
         write(*,*) "ne(2)                =",ne(2)
         write(*,*) "npack1               =",npack1
         write(*,*)
         do ms=1,ispin
            write(*,*) "ms,algorithm,mblock,nblock =",
     >                 ms,algorithm(ms),mblock(ms),nblock(ms)
         end do
      end if

      !**** lagrange multiplier test ****
      mthr = max(mthr,4)

      call lmbda_test_computations(oprint,npack1,ne,nida,nidb,
     >                      dpsi1,number_tests,mthr,
     >                      ispin,psi1,psi2,dte,lmbda,tmp,ierr)


      !if (oprint) write(*,*) "<psi2|psi2> = "
      !call check_norms(oprint,ispin,ne,nida,nidb,psi2)

      !call Parallel_Finalize()
      
 
      end

      subroutine check_norms(oprint,ispin,ne,nida,nidb,psi1)
      implicit none
      logical oprint
      integer ispin,ne(2),nida,nidb
      complex*16 psi1(*)

      integer ms,j,k,indxj,indxk,npack1,shift
      real*8  w
      npack1 = nida + nidb

      do ms=1,ispin
         do k=1,ne(ms)
         do j=1,ne(ms)
            indxj = 1 + (j-1)*npack1+(ms-1)*shift
            indxk = 1 + (k-1)*npack1+(ms-1)*shift
            call epack_cc_dot(nida,nidb,psi1(indxj),psi1(indxk),w)
            if (oprint) 
     >         write(*,'(A,I2,2I5,E18.9)') 
     >              "<psi|psi>=",ms,j,k,w
         end do
         end do
      end do

      return
      end

      subroutine lmbda_test_computations(oprint,npack1,ne,nida,nidb,
     >                      dpsi1,number_tests,mthr,
     >                      ispin,psi1,psi2,dte,lmbda,tmp,ierr)
#ifndef OPENMP_REDUCE
      USE omp_lib
#endif
      implicit none

*     **** arguments ****
      logical oprint
      integer npack1, ne(2)

*     **** psi variables ****
      !integer npack1_max,nemax
      !parameter (npack1_max=npack1,nemax=ne(1))

      integer nida,nidb
      complex*16 psi1(npack1*ne(1))
      complex*16 psi2(npack1*ne(1))
      complex*16 dpsi1(npack1*ne(1))
      real*8 lmbda(ne(1)*ne(1)*2),tmp(ne(1)*ne(1)*8)
#ifndef OPENMP_REDUCE
      real*8 thrlmbda(mthr*ne(1)*ne(1)*8)

      INTEGER(kind=omp_nest_lock_kind) reduce_lock1
      INTEGER(kind=omp_nest_lock_kind) reduce_lock2
      INTEGER(kind=omp_nest_lock_kind) reduce_lock3
      common / reduce_ffm / reduce_lock1,reduce_lock2,reduce_lock3
#else
      real*8 thrlmbda(1)
#endif

      integer taskid
      real*8  adiff
      logical notgram

!      double precision thrtreduce(mthr)
!      double precision thrtreset(mthr)
!      double precision treset,treduce
      

      integer ispin,n1(2),n2(2),ms
      integer i,j,k,n,ierr,tid,nthr,mthr
      integer number_tests
      integer ffmcount,fmfcount,mmmcount
      real*8 dte

*     **** external functions ****
      integer  Parallel_maxthreads,Parallel_index_1dblock
      external Parallel_maxthreads,Parallel_index_1dblock
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      double precision tstart,tstop,tpsi

!allocate the extra buffers for the reduce algorithm
! in thrlmbda
      call nwpw_timing_init()

!      treset  = -1.0
!      treduce = -1.0
!      do i=1,mthr
!        thrtreduce(i) = 0.0
!        thrtreset(i) = 0.0
!      end do

      call Parallel_taskid(taskid)

      ffmcount = 0
      fmfcount=0
      mmmcount=0
#ifndef OPENMP_REDUCE
      call omp_init_nest_lock(reduce_lock1)
      call omp_init_nest_lock(reduce_lock2)
      call omp_init_nest_lock(reduce_lock3)
#endif
!$OMP PARALLEL default(none)
!$OMP& shared(ffmcount,fmfcount,mmmcount)
!$OMP& firstprivate(number_tests,npack1,ispin,nida,nidb)
!$OMP& shared(dpsi1,ne,psi1,psi2,dte,ierr,taskid,adiff,notgram)
!$OMP& private(i,n,k,tid,nthr) shared(lmbda,thrlmbda,tmp)
!$OMP& shared(reduce_lock1,reduce_lock2,reduce_lock3)
      do i=1,number_tests

!!$OMP PARALLEL private(n,k,tid,nthr) 
        tid  = Parallel_threadid()
        nthr = Parallel_nthreads()
         do n=tid,(ne(1)+ne(2)-1),nthr
            do k=1,npack1
               psi2(k+n*npack1)=psi1(k+n*npack1)+dte*dpsi1(k+n*npack1)
            end do
         end do
!!$OMP END PARALLEL

! ideal thing to do is to declare the parallel region outside of this
! loop

!      tstart = omp_get_wtime()
         call psi_lmbda(ispin,ne,ne(1)+ne(2),nida,nidb,
     >                  psi1,psi2,dte,lmbda,tmp,ierr,
     >                  thrlmbda,taskid,adiff,notgram,
     >                          ffmcount,fmfcount,mmmcount)
!      tstop = omp_get_wtime()
!      tpsi = tpsi + tstop - tstart


      end do
!$OMP END PARALLEL


#ifndef OPENMP_REDUCE
      call omp_destroy_nest_lock(reduce_lock1)
      call omp_destroy_nest_lock(reduce_lock2)
      call omp_destroy_nest_lock(reduce_lock3)
#endif


!      write(*,*) "TIME ELAPSED: ",tpsi

!      do i=1,mthr
!!        write(*,*) "T",i, "Reset time",thrtreset(i), "Reduce time",
!!     >                                                  thrtreduce(i)
!        if((treset .eq. -1.0) .or. (treset .le. thrtreset(i))) then
!          treset = thrtreset(i)
!        end if
!
!        if((treduce .eq. -1.0) .or. (treduce .le. thrtreduce(i))) then
!          treduce = thrtreduce(i)
!        end if
!      end do
!
!      write(*,*) "Reset time",treset, "Reduce time",treduce

      call Parallel_barrier()
!      write(*,*) "ALIVE"
      write(*,*) "ffm count per step:",ffmcount/number_tests
      write(*,*) "fmf count per step:",fmfcount/number_tests
      write(*,*) "mmm count per step:",mmmcount/number_tests
      call nwpw_timing_print_final(oprint,number_tests)

      end




      program lmbda_test1
#ifdef MKL
      use mkl_service
#endif

      integer npack1, ne(2),np,rank
      integer numThreads

      integer  Parallel_totalthreads
      external Parallel_totalthreads,omp_debug_ids
*     **** get ne and npack from std input ****
      
      !call omp_debug_ids()

      call Parallel_Init()
      call Parallel_np(np)
      call Parallel_taskid(rank)

      numThreads = Parallel_totalthreads()
      write(*,*) "Total number of threads is",numThreads


#ifdef MKL
        call mkl_set_num_threads(numThreads)
#endif

      if(rank .eq. 0) then
          write(*,*) "Reading input parameters"
          read(*,*) npack1, ne(1), ne(2)
          npack1 = npack1 / np
          write(*,*) "Input parameters read"
      else
          npack1=0
          ne(1)=0
          ne(2)=0
      endif

      
      call Parallel_IMaxAll(npack1)
      call Parallel_IMaxAll(ne(1))
      call Parallel_IMaxAll(ne(2))
   
      
 

!      write(*,*) npack1, ne(1), ne(2)

      call lmbda_test_r(npack1,ne)

      call Parallel_Finalize()
      end program lmbda_test1
