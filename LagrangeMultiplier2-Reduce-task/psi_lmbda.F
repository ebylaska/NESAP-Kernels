*     ********************************************
*     *                                          *
*     *                psi_lmbda                 *
*     *                                          *
*     ********************************************
#define OMPTASKEXT
!#define OMPTASK
      subroutine psi_lmbda(ispin,ne,nemaxq,nida,nidb,
     >                     psi1,psi2,
     >                     dte,
     >                     lmbda,tmp,ierr,thrlmbda
     >                            ,taskid,adiff,notgram)
!     >                        ,thrtreset,thrtreduce)

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

!      double precision thrtreset(*),thrtreduce(*)
!      double precision treset,treduce
      integer dummy,wtid

      call nwpw_timing_start(3)

      call Parallel_np(np)
*::::::::::::::::::::::  Lagrangian multipliers  ::::::::::::::::::::::




      npack1 = nida+nidb
!!$OMP PARALLEL default(none) firstprivate(npack1) private(i,n,tid,nthr)
!!$OMP& private(s22,s12,st2,sa1,sa0,st1,ms1,ms2,ishift2,it) 
!!$OMP& private(shift,shift2,mb,nn,s11,s21,dummy,wtid)
!!$OMP& private(treset,treduce) firstprivate(ispin,nida,nidb,np) 
!!$OMP& shared(ne,psi1,psi2,dte,ierr,taskid,adiff,notgram) 
!!!$OMP& firstprivate(itrlmd,convg) 

#ifdef MKL
        dummy = mkl_set_num_threads_local(Parallel_nthreads())
#endif
      !treset = 0.0
      !treduce = 0.0
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


!zero out the three matrices in one operation
        call eDneall_ffm_zeroMatrix(tmp(s11),n,3*n)!,treset)

#ifdef OMPTASKEXT
!!$OMP TASKGROUP
!$OMP SINGLE

!!$OMP TASK firstprivate(n,s22,nida,nidb,shift2,shift)
!!$OMP& shared(psi2,thrlmbda,ne,tmp)
!!$OMP& depend(out:tmp(s22))
#endif
        call eDneall_ffm_sym_Multiply_reduce(psi2,psi2,
     >                                nida,nidb,ne,
     >                                tmp(s22),thrlmbda,
     >                                 shift,shift2,n)!,treset,treduce)
#ifdef OMPTASKEXT
!!$OMP END TASK
!$OMP END SINGLE 
!NOWAIT
!!$OMP TASKWAIT

!$OMP SINGLE
!!$OMP TASK firstprivate(n,s21,nida,nidb,shift2,shift)
!!$OMP& shared(psi2,psi1,thrlmbda,ne,tmp)
!!$OMP& depend(out:tmp(s21))
#endif
        call eDneall_ffm_sym_Multiply_reduce(psi2,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s21),thrlmbda,
     >                                 shift,shift2,n)!,treset,treduce)
#ifdef OMPTASKEXT
!!$OMP END TASK
!$OMP END SINGLE 
!NOWAIT
!!$OMP TASKWAIT

!$OMP SINGLE
!!$OMP TASK firstprivate(n,s11,nida,nidb,shift2,shift)
!!$OMP& shared(psi1,thrlmbda,ne,tmp)
!!$OMP& depend(out:tmp(s11))
#endif
        call eDneall_ffm_sym_Multiply_reduce(psi1,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s11),thrlmbda,
     >                                 shift,shift2,n)!,treset,treduce)
#ifdef OMPTASKEXT
!!$OMP END TASK

!$OMP END SINGLE 
!NOWAIT
!!$OMP TASKWAIT
!!$OMP END TASKGROUP
#endif

!!$OMP BARRIER

!$OMP MASTER
!$OMP TASK firstprivate(n,s11,st2,shift2,sa1,sa0,np) shared(ne,tmp)
!$OMP& depend(inout:tmp(s22)) depend(inout:tmp(s21)) 
!$OMP& depend(inout:tmp(s11))
!$OMP& final(.True.)
      call nwpw_timing_start(15)    
      if (np.gt.1) then
        !do only one big reduce and then shift data accordingly
        call Parallel_Vector_SumAll_master(3*n*n,tmp(s11),tmp(st2))

#if 0
        do elem=st2,st2+n*n-shift2
          tmp(elem)=tmp(elem+shift2)
        end do
        do elem=sa1,sa1+n*n-shift2
          tmp(elem)=tmp(elem+shift2)
        end do
        do elem=sa0,sa0+n*n-shift2
          tmp(elem)=tmp(elem+shift2)
        end do
#else
        if(shift2 .ne. 0) then
!!$OMP SINGLE
        tmp(st2:st2+n*n-shift2)=tmp(st2+shift2:st2+n*n)
        tmp(sa1:sa1+n*n-shift2)=tmp(sa1+shift2:sa1+n*n)
        tmp(sa0:sa0+n*n-shift2)=tmp(sa0+shift2:sa0+n*n)
!!$OMP END SINGLE
        end if
#endif
      end if
      call nwpw_timing_end(15)    
!$OMP END TASK
!$OMP END MASTER

  30     continue
        end do

!$OMP SINGLE
*       ***** scale the overlap matrices ****
!$OMP TASK firstprivate(mb,ispin,ne,dte,s22) shared(tmp)
!$OMP& depend(inout:tmp(s22)) final(.True.)
        call eDneall_m_scale_s22(mb,ispin,ne,dte,tmp(s22))
!$OMP END TASK

!$OMP TASK firstprivate(mb,ispin,ne,dte,s21) shared(tmp)
!$OMP& depend(inout:tmp(s21)) final(.True.)
        call eDneall_m_scale_s21(mb,ispin,ne,dte,tmp(s21))
!$OMP END TASK

!$OMP TASK firstprivate(mb,ispin,ne,dte,s11) shared(tmp)
!$OMP& depend(inout:tmp(s11)) final(.True.)
        call eDneall_m_scale_s11(mb,ispin,ne,dte,tmp(s11))
!$OMP END TASK


!$OMP TASK firstprivate(nn,s21,s12) shared(tmp) depend(in:tmp(s21))
!$OMP& depend(out:tmp(s12)) final(.True.)
        call dcopy(nn,tmp(s21),1,tmp(s12),1)
!$OMP END TASK
!$OMP TASK firstprivate(nn,s22,sa0) shared(tmp) depend(in:tmp(s22)) 
!$OMP& depend(out:tmp(sa0)) final(.True.)
        call dcopy(nn,tmp(s22),1,tmp(sa0),1)
!$OMP END TASK

!!$OMP TASK firstprivate(nn,s22,sa1) shared(tmp) depend(in:tmp(s22))
!!$OMP& depend(out:tmp(sa1)) final(.True.)
!          call dcopy(nn,tmp(s22),1,tmp(sa1),1)
!!$OMP END TASK
!$OMP END SINGLE NOWAIT

!$OMP TASKWAIT
#if 1
        do it=1,itrlmd

!$OMP SINGLE
          call dcopy(nn,tmp(s22),1,tmp(sa1),1)
!$OMP END SINGLE

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

!$OMP SINGLE
          call dcopy(nn,tmp(sa1),1,tmp(st1),1)
          call daxpy(nn,(-1.0d0),tmp(sa0),1,tmp(st1),1)
          adiff = eDneall_m_dmax(mb,ispin,ne,tmp(st1))
!          write (*,*) "adiff",adiff, "thread", tid
!$OMP END SINGLE 

!!$OMP FLUSH(adiff)

          if(adiff.lt.convg) then
                GO TO 630
          end if
          if (adiff.gt.1.0d10) then
                go to 620
          end if

!$OMP SINGLE
          call dcopy(nn,tmp(sa1),1,tmp(sa0),1)
          call dcopy(nn,tmp(s22),1,tmp(sa1),1)
!$OMP END SINGLE

        end do

  620   continue
       
        
        ierr=10 
        !write (*,*) "doing gram schmidt"
        if (taskid.eq.MASTER) then
!$OMP MASTER
          write(6,*) 
     >     'Warning: Lagrange Multiplier tolerance too high:',adiff
          write(6,*) '        +Try using a smaller time step'
          write(6,*) '        +Gram-Schmidt being performed, spin:',ms
!$OMP END MASTER
!$OMP BARRIER
        end if
c        call Dneall_f_GramSchmidt(ms,psi2,npack1)
        notgram = .false.
  630   continue

*       :::::::::::::::::  correction due to the constraint  :::::::::::::::::
        if (notgram) then
          call eDneall_fmf_Multiply(mb,ispin,ne,
     >                              psi1,nida+nidb,
     >                              tmp(sa1), dte,
     >                              psi2,1.0d0)
        end if
!$OMP SINGLE
        call eDneall_mm_Expand(mb,ne,tmp(sa1),lmbda)
!$OMP END SINGLE

#endif




  640 continue
!!$OMP END PARALLEL 

c*:::::::::::::::::  correction due to the constraint  :::::::::::::::::
      call nwpw_timing_end(3)

      return
      end

      subroutine eDneall_ffm_zeroMatrix(matrix,m,n)!,treset)
      implicit none
      integer m,n
      real*8 matrix(m,n)
!      double precision treset
!      double precision tstart,tstop
      integer k,j
!      double precision omp_get_wtime
!      external omp_get_wtime

!      tstart = omp_get_wtime()
#if 0
!$OMP SINGLE      
      matrix=0
!$OMP END SINGLE
#else
!$OMP DO schedule(static)
      do k=1,n
!$OMP SIMD
      do j=1,m
        matrix(j,k) = 0.0
      end do
!$OMP END SIMD
      end do
!$OMP END DO
#endif

!      tstop = omp_get_wtime()
!      treset = treset + tstop - tstart

      end

      subroutine eDneall_ffm_sym_Multiply_reduce(A1,A2,
     >                         nida,nidb,ne,hml,thrhml,
     >                        shift,shift2,n)!,treset,treduce)
      implicit none
      complex*16 A1(*),A2(*)
      integer nida,nidb,ispin,ne(2)
      real*8 hml(*)
      real*8 thrhml(*)

      integer n,shift,shift2
      integer nk,bk,offsetk,minchunk,nchunk
      integer itid,tid,nthr,chunk,ne1

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads,Parallel_maxthreads
      external Parallel_threadid,Parallel_nthreads,Parallel_maxthreads
!      double precision treset,treduce

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()
      nk = 2*nidb

#ifndef OMPTASKEXT
      nchunk = nthr
      chunk=tid
#else
      minchunk = 512
      nchunk = max(1,min(nthr,floor(REAL(nk)/REAL(minchunk))))
      !nchunk = max(1,floor(REAL(nk)/REAL(minchunk)))
!$OMP TASKGROUP
      do chunk=0,nchunk-1
#endif
      !compute offsets in A and B
      bk = floor(REAL(nk)/REAL(nchunk))
      offsetk = (chunk)*bk
      if(chunk==nchunk-1) bk = nk-offsetk
      !write (*,*) "tid is ",tid, nthr,nk,minchunk, "nchunk is ",nchunk

#ifdef OMPTASKEXT
!$OMP TASK firstprivate(nk,bk,offsetk,tid,shift2) private(ne1)
!$OMP& shared(A1,A2,hml,thrhml,ne)
#endif
      call nwpw_timing_start(15)    
      ne1 = ne(1)
      itid  = Parallel_threadid()
!   write (*,*) "itid is ",itid, "ne1 is ",ne1, "max threads ",
!     & Parallel_maxthreads()
         call epack_ccm_sym_dot_reduce(nida,nidb,n,
     >                     A1(shift),
     >                     A2(shift),
     >                     hml(shift2), 
     >                     thrhml((itid)*ne1*ne1*8+ shift2),
     >                     nk,bk,offsetk)!,treset,treduce)
      call nwpw_timing_end(15)    
#ifdef OMPTASKEXT
!$OMP END TASK
#endif
#ifdef OMPTASKEXT
      end do
!$OMP END TASKGROUP
#endif
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

      subroutine epack_ccm_sym_dot_reduce(nida,nidb,n,A,B,matrix,
     >                                        thrmatrix,nk,bk,offsetk)
!     >                                          ,treset,treduce)
      implicit none
      integer    nida,nidb,n
      real*8 A(*)
      real*8 B(*)
      real*8     matrix(n,n)
      real*8     thrmatrix(n,n)

*     **** local variables ****
      integer j,k,nk
      integer np,npack,npack2
      integer tid,nthr
      integer offsetk,bk,bkc
      integer offsetThread

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

!      double precision treset,treduce
!      double precision tstart,tstop
!      double precision omp_get_wtime
!      external omp_get_wtime

      call nwpw_timing_start(2)
      !call Parallel_np(np)

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      npack  = (nida+nidb)
      npack2 = 2*npack

!      nk = 2*nidb
!      bk = floor(REAL(nk)/REAL(nthr))
!      offsetk = (tid)*bk
!      if(tid==nthr-1) bk = nk-offsetk

!      write (*,*) "tid is ",tid, "vars are ",nida,nidb,n,nk,bk,offsetk

#ifndef OPENMP_REDUCE
      !tstart = omp_get_wtime()

#ifndef OMPTASK
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
#else
!!$OMP TASK depend(out:tid)
!$OMP TASKGROUP
      call dgemm_omp('T','N',n,n,bk,
     >             2.0d0,
     >             A(1+ 2*nida + offsetk),npack2,
     >             B(1+ 2*nida + offsetk),npack2,
     >             0.0d0,
     >             thrmatrix, n)

      if(tid.eq.0) then
      call dgemm_omp('T','N',n,n,2*nida,
     >             1.0d0,
     >             A,npack2,
     >             B,npack2,
     >             1.0d0,
     >             thrmatrix, n)
      end if
!$OMP END TASKGROUP
!!$OMP END TASK

!!$OMP TASK depend(in:tid)
#endif

!     THIS SHOULD BE IMPLEMENTED AS AN OPENMP REDUCTION
!     perform OMP reduction
!      tstart = omp_get_wtime()
!$OMP CRITICAL
      do k=1,n
!$OMP SIMD
      do j=1,n
        matrix(j,k) = matrix(j,k) + thrmatrix(j,k) 
      end do
!$OMP END SIMD
      end do
!$OMP END CRITICAL
!      tstop = omp_get_wtime()
!      treduce = treduce + tstop - tstart

#ifdef OMPTASK
!!$OMP END TASK
#endif

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

!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s22,ne)
!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
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
!$OMP END PARALLEL

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

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s21,ne)
!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2
!$OMP DO private(k)
        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s21(indx) = (1.0d0 - s21(indx))*0.5d0

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s21(indx)  = -s21(indx)*0.5d0
              s21(indxt) = s21(indx)
           end do
        end do
!$OMP END DO 
 30     continue
      end do
!$OMP END PARALLEL
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

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s11,ne)
!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2
!$OMP DO private(k)
        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s11(indx) = -s11(indx)*0.5d0*dte

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s11(indx)  = -s11(indx)*0.5d0*dte
              s11(indxt) = s11(indx)
           end do
        end do
!$OMP END DO 
 30     continue
      end do
!$OMP END PARALLEL
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
!$OMP PARALLEL 
               call dgemm_omp('N','N', mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                C(ishiftC), n)
!!!$OMP BARRIER
!$OMP END PARALLEL
#endif
            !*** threads and cpus ****
            else
!$OMP SINGLE
               call dcopy(n*n,0.0d0,0,tmp(shift2),1)
!$OMP END SINGLE
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
!$OMP PARALLEL 
               call dgemm_omp('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                tmp(ishiftC), n)
!!!$OMP BARRIER
!$OMP END PARALLEL
#endif
            end if
          end if
   30    continue
      end do

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

#if 1
         call DGEMM('N','N',2*(mend-mstart),n,n,
     >             (alpha),
     >             Ain(shift+mstart), 2*npack1,
     >             hml(shift2),    n,
     >             (beta),
     >             Aout(shift+mstart),2*npack1)
#else      
!$OMP PARALLEL 
         call dgemm_omp('N','N',2*(mend-mstart),n,n,
     >             (alpha),
     >             Ain(shift+mstart), 2*npack1,
     >             hml(shift2),    n,
     >             (beta),
     >             Aout(shift+mstart),2*npack1)
!$OMP END PARALLEL
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
