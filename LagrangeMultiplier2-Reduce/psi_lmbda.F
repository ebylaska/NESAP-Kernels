*     ********************************************
*     *                                          *
*     *                psi_lmbda                 *
*     *                                          *
*     ********************************************

      subroutine psi_lmbda(ispin,ne,nemaxq,nida,nidb,
     >                     psi1,psi2,
     >                     dte,
     >                     lmbda,tmp,ierr,thrlmbda,
     >                        thrtreset,thrtreduce)

#ifdef MKL
      use mkl_service
#endif
      implicit none
      integer ispin,ne(2),nemaxq,nida,nidb
      complex*16 psi1(nida+nidb,nemaxq)
      complex*16 psi2(nida+nidb,nemaxq)
      real*8     dte
      real*8     lmbda(*)
      real*8     thrlmbda(*)
      real*8     tmp(*)
      integer    ierr

      integer MASTER,tid,nthr
      parameter (MASTER=0)

*     **** local variables ****
      logical notgram
      integer taskid
      integer mb,ms,ms1,ms2,it
      integer nn,i
      integer s11,s12,s21,s22,st1,st2,sa1,sa0
      real*8  adiff

*     ::::  iteration limit and tolerence for non-liner equations  ::::
      integer itrlmd
      real*8  convg
      parameter (itrlmd=120, convg=1.0d-15)

      integer shift,shift2,ishift2
      integer n,npack1

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads,Parallel_totalthreads
      external Parallel_threadid,Parallel_nthreads,Parallel_totalthreads

      integer np

      real*8   eDneall_m_dmax
      external eDneall_m_dmax

      double precision thrtreset(*),thrtreduce(*)
      double precision treset,treduce
      integer counter, dummy

      call nwpw_timing_start(3)

      call Parallel_np(np)
*::::::::::::::::::::::  Lagrangian multipliers  ::::::::::::::::::::::

      !call omp_debug_ids()



      npack1 = nida+nidb
!$OMP PARALLEL firstprivate(npack1), private(i,n,tid,nthr,shift,shift2),&
!$OMP& private(treset,treduce) proc_bind(spread)


#ifdef MKL
!reset the mkl_num_threads by setting it to an invalid value
        dummy = mkl_set_num_threads_local(Parallel_nthreads())
#endif
      counter = 0
      treset = 0.0
      treduce = 0.0
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()
      DO 640 mb=1,ispin
        notgram=.true.
        IF(ne(mb).le.0) GO TO 640

        !call Dneall_m_size(ms,nn)
        nn = ne(mb)*ne(mb)
        s11  = 0*nn + 1
        s21  = 1*nn + 1
        s22  = 2*nn + 1
        s12  = 3*nn + 1
        st2  = 4*nn + 1
        sa1  = 5*nn + 1
        sa0  = 6*nn + 1
        st1  = 7*nn + 1

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if



      do ms=ms1,ms2
         shift  = 1 + (ms-1)*ne(1)*npack1
         shift2 = 1 + (ms-1)*ishift2
         n     = ne(ms)
         if (n.le.0) go to 30

         counter = counter+3

!zero out the three matrices in one operation
!$OMP SINGLE      
        call eDneall_ffm_zeroMatrix(tmp(s11),n,3*n,treset)
!$OMP END SINGLE

!!$OMP SINGLE      
!        call eDneall_ffm_zeroMatrix(tmp(s22),n,n,treset)
!!$OMP END SINGLE
        call eDneall_ffm_sym_Multiply_reduce(psi2,psi2,
     >                                nida,nidb,ne,
     >                                tmp(s22),tmp(sa0),thrlmbda,
     >                                 shift,shift2,n,treset,treduce)

!!$OMP SINGLE      
!        call eDneall_ffm_zeroMatrix(tmp(s21),n,n,treset)
!!$OMP END SINGLE       
        call eDneall_ffm_sym_Multiply_reduce(psi2,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s21),tmp(sa1),thrlmbda,
     >                                 shift,shift2,n,treset,treduce)

!!$OMP SINGLE      
!        call eDneall_ffm_zeroMatrix(tmp(s11),n,n,treset)
!!$OMP END SINGLE       
        call eDneall_ffm_sym_Multiply_reduce(psi1,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s11),tmp(st1),thrlmbda,
     >                                 shift,shift2,n,treset,treduce)

!$OMP BARRIER
      call nwpw_timing_start(15)    
      if (np.gt.1) then
        !do only one big reduce and then shift data accordingly
        call Parallel_Vector_SumAll(3*n*n,tmp(s11),tmp(st2))
!$OMP SINGLE
        tmp(st2:st2+n*n-shift2)=tmp(st2+shift2:st2+n*n)
        tmp(sa1:sa1+n*n-shift2)=tmp(sa1+shift2:sa1+n*n)
        tmp(sa0:sa0+n*n-shift2)=tmp(sa0+shift2:sa0+n*n)
!$OMP END SINGLE
!        call Parallel_Vector_SumAll(n*n,tmp(s11+shift2),tmp(st2))
!        call Parallel_Vector_SumAll(n*n,tmp(s21+shift2),tmp(sa1))
!        call Parallel_Vector_SumAll(n*n,tmp(s22+shift2),tmp(sa0))
      end if
      call nwpw_timing_end(15)    

  30     continue
        end do

      thrtreset(tid+1) = thrtreset(tid+1) + treset
      thrtreduce(tid+1) = thrtreduce(tid+1) + treduce

*       ***** scale the overlap matrices ****
        call eDneall_m_scale_s22(mb,ispin,ne,dte,tmp(s22))
        call eDneall_m_scale_s21(mb,ispin,ne,dte,tmp(s21))
        call eDneall_m_scale_s11(mb,ispin,ne,dte,tmp(s11))

  640 continue
!$OMP END PARALLEL 


        

#ifdef MKL
        dummy = mkl_set_num_threads_local(0)
#endif

      DO 650 mb=1,ispin
        notgram=.true.
        IF(ne(mb).le.0) GO TO 650

        !call Dneall_m_size(ms,nn)
        nn = ne(mb)*ne(mb)
        s11  = 0*nn + 1
        s21  = 1*nn + 1
        s22  = 2*nn + 1
        s12  = 3*nn + 1
        st2  = 4*nn + 1
        sa1  = 5*nn + 1
        sa0  = 6*nn + 1
        st1  = 7*nn + 1



!$OMP BARRIER
!$OMP MASTER
        call dcopy(nn,tmp(s21),1,tmp(s12),1)
        call dcopy(nn,tmp(s22),1,tmp(sa0),1)
!$OMP END MASTER

        do it=1,itrlmd
!$OMP BARRIER
!$OMP MASTER 
          call dcopy(nn,tmp(s22),1,tmp(sa1),1)
!$OMP END MASTER
!$OMP BARRIER

          call eDneall_mmm_Multiply(mb,ispin,ne,
     >                              tmp(s21),tmp(sa0),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))
          call eDneall_mmm_Multiply(mb,ispin,ne,
     >                              tmp(sa0),tmp(s12),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))
          call eDneall_mmm_Multiply(mb,ispin,ne,
     >                              tmp(s11),tmp(sa0),1.0d0,
     >                              tmp(st1),0.0d0,tmp(st2))
          call eDneall_mmm_Multiply(mb,ispin,ne,
     >                              tmp(sa0),tmp(st1),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))

!$OMP MASTER 
          call dcopy(nn,tmp(sa1),1,tmp(st1),1)
          call daxpy(nn,(-1.0d0),tmp(sa0),1,tmp(st1),1)
!$OMP END MASTER
!$OMP BARRIER

          adiff = eDneall_m_dmax(mb,ispin,ne,tmp(st1))
          if(adiff.lt.convg) GO TO 630
          if (adiff.gt.1.0d10) go to 620
!$OMP MASTER 
          call dcopy(nn,tmp(sa1),1,tmp(sa0),1)
!$OMP END MASTER
        end do

  620   continue
        ierr=10 
        call Parallel_taskid(taskid)
        if (taskid.eq.MASTER) then
          write(6,*) 
     >     'Warning: Lagrange Multiplier tolerance too high:',adiff
          write(6,*) '        +Try using a smaller time step'
          write(6,*) '        +Gram-Schmidt being performed, spin:',ms
        end if
c        call Dneall_f_GramSchmidt(ms,psi2,npack1)
        notgram = .false.

  630   continue

*       :::::::::::::::::  correction due to the constraint  :::::::::::::::::
        if (notgram)
     >     call eDneall_fmf_Multiply(mb,ispin,ne,
     >                              psi1,nida+nidb,
     >                              tmp(sa1), dte,
     >                              psi2,1.0d0)

!$OMP BARRIER
!$OMP MASTER 
        call eDneall_mm_Expand(mb,ne,tmp(sa1),lmbda)
!$OMP END MASTER 
  650 continue
!      end do
!!!$OMP END PARALLEL



c*:::::::::::::::::  correction due to the constraint  :::::::::::::::::
      call nwpw_timing_end(3)

!$OMP BARRIER
      return
      end

      subroutine eDneall_ffm_zeroMatrix(matrix,m,n,treset)
      implicit none
      integer m,n
      real*8 matrix(m,n)
      double precision treset
      double precision tstart,tstop
      integer k,j
      double precision omp_get_wtime
      external omp_get_wtime

      tstart = omp_get_wtime()
      matrix=0
!!!!!$OMP DO
!      do k=1,n
!!$OMP SIMD
!      do j=1,n
!        matrix(j,k) = 0.0
!      end do
!!$OMP END SIMD
!      end do
!!!!$OMP END DO

      tstop = omp_get_wtime()
      treset = treset + tstop - tstart

      end

      subroutine eDneall_ffm_sym_Multiply_reduce(A1,A2,
     >                         nida,nidb,ne,hml,tmp,thrhml,
     >                        shift,shift2,n,treset,treduce)
      implicit none
      complex*16 A1(*),A2(*)
      integer nida,nidb,ispin,ne(2)
      real*8 hml(*),tmp(*)
      real*8 thrhml(*)

      integer n,shift,shift2
      integer tid

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads
      double precision treset,treduce

      call nwpw_timing_start(15)    
      tid  = Parallel_threadid()
         call epack_ccm_sym_dot_reduce(nida,nidb,n,
     >                     A1(shift),
     >                     A2(shift),
     >                     hml(shift2), tmp,
     >                     thrhml((tid)*ne(1)*ne(1)*8+ shift2),
     >                            treset,treduce)


      call nwpw_timing_end(15)    
      return
      end


*     ***********************************
*     *                                 *
*     *         epack_ccm_sym_dot       *       
*     *                                 *
*     ***********************************

      subroutine epack_ccm_sym_dot(nida,nidb,n,A,B,matrix,tmp)
      implicit none
      integer    nida,nidb,n
      complex*16 A(*)
      complex*16 B(*)
      real*8     matrix(n,n)
      real*8     tmp(*)

*     **** local variables ****
      integer j,k
      integer np,npack,npack2
      integer tid,nthr
      integer offsetk,bk

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads


      call nwpw_timing_start(2)
      call Parallel_np(np)

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      npack  = (nida+nidb)
      npack2 = 2*npack


      do k=tid+1,n,nthr
        call DGEMM('T','N',k,1,2*nida,
     >             1.0d0,
     >             A,npack2,
     >             B(1+(k-1)*npack),npack2,
     >             0.0d0,
     >             matrix(1,k),k)
        call DGEMM('T','N',k,1,npack2-2*nida,
     >             2.0d0,
     >             A(1+nida),npack2,
     >             B(1+nida+(k-1)*npack),npack2,
     >             1.0d0,
     >             matrix(1,k),k)
      end do

!$OMP BARRIER
      do k=tid+1,n,nthr
      do j=k+1,n
        matrix(j,k) = matrix(k,j)
      end do
      end do

      if (np.gt.1) call Parallel_Vector_SumAll(n*n,matrix,tmp)

      call nwpw_timing_end(2)
      return
      end

      subroutine omp_debug_ids()
      USE ISO_FORTRAN_ENV, ONLY : ERROR_UNIT
      implicit none
      integer tid,nthr,level,plevel,gtid
      integer ptid,pnthr,i
      integer  Parallel_threadid,omp_get_num_threads
      external Parallel_threadid,omp_get_num_threads
      integer  omp_get_active_level
      external omp_get_active_level
!$OMP PARALLEL private(tid,nthr,ptid,pnthr,level,plevel,gtid)
!$OMP& proc_bind(spread)
        level = omp_get_active_level()
        tid  = Parallel_threadid()
        nthr = omp_get_num_threads()
        do i=0,nthr-1
        if(tid .eq. i) then
        write(ERROR_UNIT,*) "level is",level,"tid is",tid,
     >                                       ", nthr is",nthr
!$OMP PARALLEL private(ptid,pnthr,plevel,gtid) proc_bind(close)
        plevel = omp_get_active_level()
        ptid  = Parallel_threadid()
        pnthr = omp_get_num_threads()

        if(ptid.eq.0) then
           gtid=tid
        else
           gtid=nthr+tid*(pnthr-1)+ptid-1 
        end if

        write(ERROR_UNIT,*) "      ","level is",plevel,"ptid is",ptid,
     >                              ", pnthr is",pnthr,", gtid is",gtid
!$OMP END PARALLEL
        end if
!$OMP BARRIER
        end do
!$OMP END PARALLEL
      end


      subroutine epack_ccm_sym_dot_reduce(nida,nidb,n,A,B,matrix,tmp,
     >                                                    thrmatrix,
     >                                          treset,treduce)
      implicit none
      integer    nida,nidb,n
      real*8 A(*)
      real*8 B(*)
      real*8     matrix(n,n)
      real*8     thrmatrix(n,n)
      real*8     tmp(*)

*     **** local variables ****
      integer j,k,nk
      integer np,npack,npack2
      integer tid,nthr
      integer offsetk,bk,bkc
      integer offsetThread

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      double precision treset,treduce
      double precision tstart,tstop
      double precision omp_get_wtime
      external omp_get_wtime

      call nwpw_timing_start(2)
      call Parallel_np(np)

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      npack  = (nida+nidb)
      npack2 = 2*npack

!!$OMP SINGLE
!      call eDneall_ffm_zeroMatrix(matrix,n,n,treset)
!!$OMP END SINGLE

!      tstart = omp_get_wtime()
!!$OMP SINGLE
!!!$OMP DO SIMD
!      do k=1,n
!!$OMP SIMD
!      do j=1,n
!        matrix(j,k) = 0.0
!      end do
!!$OMP END SIMD
!      end do
!!!$OMP END DO SIMD nowait
!!$OMP END SINGLE
!      tstop = omp_get_wtime()
!      treset = treset + tstop - tstart

#ifndef OPENMP_REDUCE

!      tstart = omp_get_wtime()
!!!$OMP BARRIER
!!!!!$OMP SINGLE
!!!!!!$OMP DO SIMD
!!!!      do k=1,n
!!!!!$OMP SIMD
!!!!      do j=1,n
!!!!        matrix(j,k) = 0.0
!!!!      end do
!!!!!$OMP END SIMD
!!!!      end do
!!!!!!$OMP END DO SIMD nowait
!!!!!$OMP END SINGLE
!
!!$OMP DO collapse(2)
!      do k=1,n
!      do j=1,n
!        matrix(j,k) = 0.0
!      end do
!      end do
!!$OMP END DO
!
!
!
!      tstop = omp_get_wtime()
!      treset = treset + tstop - tstart

!      call eDneall_ffm_zeroMatrix(matrix,n,treset)

      !tstart = omp_get_wtime()
      nk = npack2 - 2*nida
      bk = floor(REAL(nk)/REAL(nthr))
      offsetk = (tid)*bk
      if(tid==nthr-1) bk = nk-offsetk

      call DGEMM('T','N',n,n,bk,
     >             2.0d0,
     >             A(1+ 2*nida + offsetk),npack2,
     >             B(1+ 2*nida + offsetk),npack2,
     >             0.0d0,
     >             thrmatrix, n)
      if(tid.eq.0) then
      call DGEMM('T','N',n,n,2*nida,
     >             1.0d0,
     >             A,npack2,
     >             B,npack2,
     >             1.0d0,
     >             thrmatrix, n)
      end if

!     THIS SHOULD BE IMPLEMENTED AS AN OPENMP REDUCTION
!     perform OMP reduction
      tstart = omp_get_wtime()
!$OMP CRITICAL
      do k=1,n
!$OMP SIMD
      do j=1,n
        matrix(j,k) = matrix(j,k) + thrmatrix(j,k) 
      end do
!$OMP END SIMD
      end do
!$OMP END CRITICAL
      tstop = omp_get_wtime()
      treduce = treduce + tstop - tstart

#else

!$OMP DO schedule(static,1) REDUCTION(+:matrix)
      do tid=1,nthr
      nk = npack2 - 2*nida
      bk = floor(REAL(nk)/REAL(nthr))
      offsetk = (tid)*bk
      if(tid==nthr-1) bk = nk-offsetk

      call DGEMM('T','N',n,n,bk,
     >             2.0d0,
     >             A(1+ 2*nida + offsetk),npack2,
     >             B(1+ 2*nida + offsetk),npack2,
     >             1.0d0,
     >             matrix, n)
      if(tid.eq.0) then
      call DGEMM('T','N',n,n,2*nida,
     >             1.0d0,
     >             A,npack2,
     >             B,npack2,
     >             1.0d0,
     >             matrix, n)
      end if
      end do
!$OMP END DO


#endif




      call nwpw_timing_end(2)
      return
      end

















c     ****************************************
c     *                                      *
c     *        Dneall_m_scale_s22           *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s22(mb,ispin,ne,dte,s22)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s22(*)
        

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

!$OMP DO private(k)
        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s22(indx) = (1.0d0 - s22(indx))*0.5d0/dte

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s22(indx)  = -s22(indx)*0.5d0/dte
              s22(indxt) = s22(indx)
           end do
        end do
!$OMP END DO

 30     continue
      end do

      return
      end



c     ****************************************
c     *                                      *
c     *        Dneall_m_scale_s21            *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s21(mb,ispin,ne,dte,s21)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s21(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt
      integer tid,nthr

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

        do k=tid+1,ne(ms),nthr
           indx = k + (k-1)*ne(ms) + shift2
           s21(indx) = (1.0d0 - s21(indx))*0.5d0

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s21(indx)  = -s21(indx)*0.5d0
              s21(indxt) = s21(indx)
           end do
        end do

 30     continue
      end do
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_m_scale_s11           *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s11(mb,ispin,ne,dte,s11)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s11(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt
      integer tid,nthr

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

        do k=tid+1,ne(ms),nthr
           indx = k + (k-1)*ne(ms) + shift2
           s11(indx) = -s11(indx)*0.5d0*dte

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s11(indx)  = -s11(indx)*0.5d0*dte
              s11(indxt) = s11(indx)
           end do
        end do

 30     continue
      end do
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_mmm_Multiply          *
c     *                                      *
c     ****************************************

      subroutine eDneall_mmm_Multiply(mb,ispin,ne,A,B,alpha,C,beta,tmp)
      implicit none
      integer mb,ispin,ne(2)
      real*8 A(*),B(*),C(*)
      real*8 alpha,beta
      real*8 tmp(*)

*     **** local variables ****
      integer MASTER,taskid,np,tid,nthr
      parameter (MASTER=0)
      integer ms,ms1,ms2,n,shift2,ishift2
      integer mstart,mend,nstart,nend,i,j,pindx
      integer ishiftA,ishiftB,ishiftC

*     **** matrix_blocking common block ****
      integer mblock(2),nblock(2),algorithm(2)
      common /matrix_blocking/ mblock,nblock,algorithm

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads
      integer  Parallel_index_1dblock
      external Parallel_index_1dblock

      call Parallel_taskid(taskid)
      call Parallel_np(np)
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
         n     = ne(ms)
         if (n.le.0) go to 30
         shift2 = 1 + (ms-1)*ishift2

         !*** completely serial ****
         if (algorithm(ms).lt.0) then
            if (tid.eq.MASTER)
     >         call DGEMM('N','N',n,n,n,
     >                alpha,
     >                A(shift2), n,
     >                B(shift2), n,
     >                beta,
     >                C(shift2), n)
          else
            pindx = tid + taskid*nthr
            i = mod(pindx,mblock(ms))
            j = (pindx-i)/mblock(ms)
            mstart = Parallel_index_1dblock(n,mblock(ms),i)
            mend   = Parallel_index_1dblock(n,mblock(ms),i+1)
            nstart = Parallel_index_1dblock(n,nblock(ms),j)
            nend   = Parallel_index_1dblock(n,nblock(ms),j+1)
            ishiftA = shift2 + mstart 
            ishiftB = shift2 + nstart*n
            ishiftC = shift2 + mstart + nstart*n

            !*** just threaded ****
            if (algorithm(ms).lt.1) then
#if 1
               call DGEMM('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                C(ishiftC), n)
#else
               call omp_dgemm('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                C(ishiftC), n, BLK_MMM)
!$OMP BARRIER
#endif
            !*** threads and cpus ****
            else
!$OMP MASTER
               call dcopy(n*n,0.0d0,0,tmp(shift2),1)
!$OMP END MASTER
!$OMP BARRIER
               call dlacpy('G',(mend-mstart),(nend-nstart),
     >                     C(ishiftC),n,tmp(ishiftC),n)
#if 1
               call DGEMM('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                tmp(ishiftC), n)
#else
               call omp_dgemm('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                tmp(ishiftC), n,BLK_MMM)
!$OMP BARRIER
#endif
            end if
          end if
   30    continue
      end do

!$OMP BARRIER
!$OMP MASTER
      if (mb.eq.0) then
         if ((algorithm(1).lt.1).and.(algorithm(2).lt.1)) then
            call Parallel_Brdcst_values(MASTER,
     >                   ne(1)*ne(1)+ne(2)*ne(2),C)

         else if (algorithm(1).lt.1) then
            call Parallel_Brdcst_values(MASTER,
     >                   ne(1)*ne(1),C)
            call Parallel_Vector_SumAll2(
     >                  ne(2)*ne(2),tmp(ne(1)*ne(1)+1),C(ne(1)*ne(1)+1))

         else if (algorithm(2).lt.1) then
            call Parallel_Vector_SumAll2(
     >                   ne(1)*ne(1),tmp,C)
            call Parallel_Brdcst_values(MASTER,
     >                   ne(2)*ne(2),C(ne(1)*ne(1)+1))

         else 
            call Parallel_Vector_SumAll2(
     >                   ne(1)*ne(1)+ne(2)*ne(2),tmp,C)
         end if
      else
         if (algorithm(mb).lt.1) then
            call Parallel_Brdcst_values(MASTER,ne(mb)*ne(mb),C)
         else
            call Parallel_Vector_SumAll2(ne(mb)*ne(mb),tmp,C)
         end if
      end if
!$OMP END MASTER
!$OMP BARRIER
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_m_dmax                 *
c     *                                      *
c     ****************************************
         
      double precision function eDneall_m_dmax(mb,ispin,ne,A)
      implicit none
      integer mb,ispin,ne(2)
      real*8 A(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2
      double precision adiff1, adiff2
           
      integer  idamax
      external idamax

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      adiff1 = 0.0d0
      adiff2 = 0.0d0
      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = 1 + (ms-1)*ishift2

        adiff1 = adiff2
        adiff2 = A(shift2-1+idamax(ne(ms)*ne(ms),A(shift2),1))
        adiff2 = dabs(adiff2)
        if (adiff2.gt.adiff1) adiff1 = adiff2
 30     continue
      end do

      eDneall_m_dmax = adiff1
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_fmf_Multiply          *
c     *                                      *
c     ****************************************
          
      subroutine eDneall_fmf_Multiply(mb,ispin,ne,Ain,npack1,
     >                                hml,alpha,
     >                                Aout,beta)
      implicit none
      integer    mb,ispin,ne(2)
      complex*16 Ain(*)
      integer    npack1
      real*8     hml(*)
      real*8     alpha
      complex*16 Aout(*)
      real*8     beta
        

*     **** local variables ****
      integer tid,nthr,mstart,mend
      integer ms,ms1,ms2,n,shift,shift2,shift3,ishift2,ishift3

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads
      integer  Parallel_index_1dblock
      external Parallel_index_1dblock

      call nwpw_timing_start(16)    
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()
      mstart = Parallel_index_1dblock(npack1,nthr,tid)
      mend   = Parallel_index_1dblock(npack1,nthr,tid+1)

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
         n     = ne(ms)
         if (n.le.0) go to 30
         shift  = 1 + (ms-1)*ne(1)*npack1
         shift2 = 1 + (ms-1)*ishift2

#if 0
         call DGEMM('N','N',2*(mend-mstart),n,n,
     >             (alpha),
     >             Ain(shift+mstart), 2*npack1,
     >             hml(shift2),    n,
     >             (beta),
     >             Aout(shift+mstart),2*npack1)
#else      
         call omp_dgemm('N','N',2*(mend-mstart),n,n,
     >             (alpha),
     >             Ain(shift+mstart), 2*npack1,
     >             hml(shift2),    n,
     >             (beta),
     >             Aout(shift+mstart),2*npack1,BLK_FMF)
#endif
   30    continue
      end do

      call nwpw_timing_end(16)
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_mm_Expand             *
c     *                                      *
c     ****************************************

      subroutine eDneall_mm_Expand(mb,ne,A,A0)
      implicit none
      integer mb,ne(2)
      real*8 A(*),A0(*)


*     **** local variables ****
      integer shift2,nn

      shift2 = 1
      if (mb.eq.0) then
         nn     = ne(1)*ne(1) + ne(2)*ne(2)
         shift2 = 1
      else if (mb.eq.1) then
         nn     = ne(1)*ne(1)
         shift2 = 1
      else if (mb.eq.2) then
         nn     = ne(2)*ne(2)
         shift2 = 1+ne(1)*ne(1)
      end if


      call dcopy(nn,A,1,A0(shift2),1)
      return
      end


*> \brief \b DLACPY copies all or part of one two-dimensional array to another.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download DLACPY + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlacpy.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlacpy.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlacpy.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DLACPY( UPLO, M, N, A, LDA, B, LDB )
* 
*       .. Scalar Arguments ..
*       CHARACTER          UPLO
*       INTEGER            LDA, LDB, M, N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DLACPY copies all or part of a two-dimensional matrix A to another
*> matrix B.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] UPLO
*> \verbatim
*>          UPLO is CHARACTER*1
*>          Specifies the part of the matrix A to be copied to B.
*>          = 'U':      Upper triangular part
*>          = 'L':      Lower triangular part
*>          Otherwise:  All of the matrix A
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix A.  M >= 0.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrix A.  N >= 0.
*> \endverbatim
*>
*> \param[in] A
*> \verbatim
*>          A is DOUBLE PRECISION array, dimension (LDA,N)
*>          The m by n matrix A.  If UPLO = 'U', only the upper triangle
*>          or trapezoid is accessed; if UPLO = 'L', only the lower
*>          triangle or trapezoid is accessed.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A.  LDA >= max(1,M).
*> \endverbatim
*>
*> \param[out] B
*> \verbatim
*>          B is DOUBLE PRECISION array, dimension (LDB,N)
*>          On exit, B = A in the locations specified by UPLO.
*> \endverbatim
*>
*> \param[in] LDB
*> \verbatim
*>          LDB is INTEGER
*>          The leading dimension of the array B.  LDB >= max(1,M).
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date September 2012
*
*> \ingroup auxOTHERauxiliary
*
*  =====================================================================
      SUBROUTINE DLACPY( UPLO, M, N, A, LDA, B, LDB )
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            LDA, LDB, M, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, J
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MIN
*     ..
*     .. Executable Statements ..
*
      IF( LSAME( UPLO, 'U' ) ) THEN
         DO 20 J = 1, N
            DO 10 I = 1, MIN( J, M )
               B( I, J ) = A( I, J )
   10       CONTINUE
   20    CONTINUE
      ELSE IF( LSAME( UPLO, 'L' ) ) THEN
         DO 40 J = 1, N
            DO 30 I = J, M
               B( I, J ) = A( I, J )
   30       CONTINUE
   40    CONTINUE
      ELSE
         DO 60 J = 1, N
            DO 50 I = 1, M
               B( I, J ) = A( I, J )
   50       CONTINUE
   60    CONTINUE
      END IF
      RETURN
*
*     End of DLACPY
*
      END
