*     ********************************************
*     *                                          *
*     *                psi_lmbda                 *
*     *                                          *
*     ********************************************
      subroutine psi_lmbda(ispin,ne,nemaxq,nida,nidb,
     >                     psi1,psi2,
     >                     dte,
     >                     lmbda,tmp,ierr,thrlmbda
     >                            ,taskid,adiff,notgram,
     >                             ffmcount,fmfcount,mmmcount)
!     >                        ,thrtreset,thrtreduce)
      use itt_sde_fortran
      USE omp_lib
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
      integer ffmcount,fmfcount,mmmcount
      parameter (MASTER=0)

*     **** local variables ****
      logical notgram
      integer taskid
      integer mb,ms,ms1,ms2,it
      integer nn,i
      integer s11,s12,s21,s22,st1,st2,sa1,sa0
      real*8  adiff

*     **** matrix_blocking common block ****
      integer mblock(2),nblock(2),algorithm(2)
      common /matrix_blocking/ mblock,nblock,algorithm

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
      integer dummy,wtid,wnthr

      INTEGER(kind=omp_nest_lock_kind) reduce_lock1
      INTEGER(kind=omp_nest_lock_kind) reduce_lock2
      INTEGER(kind=omp_nest_lock_kind) reduce_lock3
      common / reduce_ffm / reduce_lock1,reduce_lock2,reduce_lock3
      call nwpw_timing_start(3)

      call Parallel_np(np)
*::::::::::::::::::::::  Lagrangian multipliers  ::::::::::::::::::::::


      call start_collection()


      npack1 = nida+nidb

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

!$OMP MASTER
      ffmcount = ffmcount+3
!$OMP END MASTER

      call nwpw_timing_start_thr(15)    
      call nwpw_timing_start_thr(2)
        wnthr = max(1,(nthr - mod(nthr,3))/3)
        call eDneall_ffm_sym_Multiply_reduce_concur(psi2,psi2,
     >                                nida,nidb,ne,
     >                                tmp(s22),thrlmbda,
     >                       shift,shift2,n,0,wnthr,reduce_lock1)
        call eDneall_ffm_sym_Multiply_reduce_concur(psi2,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s21),thrlmbda,
     >             shift,shift2,n,min(nthr-1,wnthr),wnthr,reduce_lock2)
        call eDneall_ffm_sym_Multiply_reduce_concur(psi1,psi1,
     >                                nida,nidb,ne,
     >                                tmp(s11),thrlmbda,shift,shift2,n,
     >            min(nthr-1,2*wnthr),max(1,nthr-2*wnthr),reduce_lock3)
      call nwpw_timing_end_thr(2)
      call nwpw_timing_end_thr(15)

      if (np.gt.1) then
      call nwpw_timing_start_thr(48) 
!$OMP BARRIER
      call nwpw_timing_end_thr(48) 
      call nwpw_timing_start_thr(15)    
!$OMP MASTER
        !do only one big reduce and then shift data accordingly
        call Parallel_Vector_SumAll_master(3*n*n,tmp(s11),tmp(st2))
        if(shift2 .ne. 0) then
          tmp(st2:st2+n*n-shift2)=tmp(st2+shift2:st2+n*n)
          tmp(sa1:sa1+n*n-shift2)=tmp(sa1+shift2:sa1+n*n)
          tmp(sa0:sa0+n*n-shift2)=tmp(sa0+shift2:sa0+n*n)
        end if
!$OMP END MASTER
      call nwpw_timing_end_thr(15)    
      end if

      call nwpw_timing_start_thr(48) 
!$OMP BARRIER
      call nwpw_timing_end_thr(48) 
  30     continue
        end do


!&OMP SECTIONS
!&OMP SECTION
        call eDneall_m_scale_s22(mb,ispin,ne,dte,tmp(s22))
!&OMP END SECTION
!&OMP SECTION
        call eDneall_m_scale_s21(mb,ispin,ne,dte,tmp(s21))
!&OMP END SECTION
!&OMP SECTION
        call eDneall_m_scale_s11(mb,ispin,ne,dte,tmp(s11))
!&OMP END SECTION

!&OMP END SECTIONS

!$OMP SINGLE
        call dcopy(nn,tmp(s22),1,tmp(sa0),1)
!$OMP END SINGLE NOWAIT
!$OMP SINGLE
        call dcopy(nn,tmp(s21),1,tmp(s12),1)
!$OMP END SINGLE


        do it=1,itrlmd
!$OMP SINGLE
          ! sa1 = s22
          call dcopy(nn,tmp(s22),1,tmp(sa1),1)
!$OMP END SINGLE
      call nwpw_timing_start(49)    
          wnthr = 1!(nthr - mod(nthr,3))/3

      do ms=ms1,ms2
          n     = ne(ms)
          if (n.gt.0) then
          shift2 = 1 + (ms-1)*ishift2


          if (algorithm(ms).gt.1) then
!$OMP SINGLE
                 call dcopy(n*n,0.0d0,0,thrlmbda((0)*nn+ 1),1)
!$OMP END SINGLE NOWAIT

!$OMP SINGLE
                 call dcopy(n*n,0.0d0,0,thrlmbda((1)*nn+ 1),1)
!$OMP END SINGLE NOWAIT

!$OMP SINGLE
                 call dcopy(n*n,0.0d0,0,thrlmbda((2)*nn+ 1),1)
!$OMP END SINGLE NOWAIT

!$OMP BARRIER
          end if

          call eDneall_mmm_Multiply_concur(mb,ispin,ne,
     >                              tmp(s21),tmp(sa0),1.0d0,
     >                              tmp(sa1),1.0d0,thrlmbda((0)*nn+ 1),
     >                                              ms,shift2,n,0,wnthr)
          call eDneall_mmm_Multiply_concur(mb,ispin,ne,
     >                              tmp(sa0),tmp(s12),1.0d0,
     >                   thrlmbda((3)*nn+ 1),0.0d0,thrlmbda((1)*nn+ 1),
     >                             ms,shift2,n,min(nthr-1,wnthr),wnthr)
          call eDneall_mmm_Multiply_concur(mb,ispin,ne,
     >                              tmp(s11),tmp(sa0),1.0d0,
     >                              tmp(st1),0.0d0,thrlmbda((2)*nn+ 1),
     >                           ms,shift2,n,min(2*wnthr,nthr-1),wnthr)

!$OMP MASTER
      mmmcount = mmmcount+3
!$OMP END MASTER

      call nwpw_timing_start_thr(48)    
!$OMP BARRIER
      call nwpw_timing_end_thr(48)    
        end if
      end do

          !do the mpi reduce here

          call eDneall_mmm_Multiply_concur_red(mb,ispin,ne,
     >                              tmp(s21),tmp(sa0),1.0d0,
     >                              tmp(sa1),1.0d0,thrlmbda((0)*nn+ 1),
     >                                              ms,shift2,n,0,wnthr)
          call eDneall_mmm_Multiply_concur_red(mb,ispin,ne,
     >                              tmp(sa0),tmp(s12),1.0d0,
     >                   thrlmbda((3)*nn+ 1),0.0d0,thrlmbda((1)*nn+ 1),
     >                              ms,shift2,n,min(nthr-1,wnthr),wnthr)
          call eDneall_mmm_Multiply_concur_red(mb,ispin,ne,
     >                              tmp(s11),tmp(sa0),1.0d0,
     >                              tmp(st1),0.0d0,thrlmbda((2)*nn+ 1),
     >                            ms,shift2,n,min(nthr-1,2*wnthr),wnthr)

!$OMP SINGLE
      call daxpy(nn,(1.0d0),thrlmbda(3*nn+1),1,tmp(sa1),1)
!$OMP END SINGLE 

      wnthr = 1
      do ms=ms1,ms2
          n     = ne(ms)
          if (n.gt.0) then
          shift2 = 1 + (ms-1)*ishift2


          if (algorithm(ms).gt.1) then
!$OMP SINGLE
                 call dcopy(n*n,0.0d0,0,tmp(st2),1)
!$OMP END SINGLE
          end if


!      call nwpw_timing_start_thr(48) 
!!$OMP BARRIER
!      call nwpw_timing_end_thr(48) 
          
          call eDneall_mmm_Multiply_concur(mb,ispin,ne,
     >                              tmp(sa0),tmp(st1),1.0d0,
     >            tmp(sa1),1.0d0,tmp(st2),ms,shift2,n, 0,wnthr)
!$OMP MASTER
      mmmcount = mmmcount+1
!$OMP END MASTER
      call nwpw_timing_start_thr(48)    
!$OMP BARRIER
      call nwpw_timing_end_thr(48)    
        end if
      end do
      !mpi reduce ?
      call eDneall_mmm_Multiply_concur_red(mb,ispin,ne,
     >                              tmp(sa0),tmp(st1),1.0d0,
     >            tmp(sa1),1.0d0,tmp(st2),ms,shift2, 0,wnthr)

      call nwpw_timing_start_thr(48) 
!$OMP BARRIER
      call nwpw_timing_end_thr(48) 
      call nwpw_timing_end(49)    

!$OMP SINGLE
          !st1 = sa1
          call dcopy(nn,tmp(sa1),1,tmp(st1),1)
          !  st1 -= sa0
          call daxpy(nn,(-1.0d0),tmp(sa0),1,tmp(st1),1)
          adiff = eDneall_m_dmax(mb,ispin,ne,tmp(st1))
!$OMP END SINGLE 

          if(adiff.lt.convg) then
                GO TO 630
          end if
          if (adiff.gt.1.0d10) then
                go to 620
          end if

!$OMP SINGLE
          !sa0 = sa1
          call dcopy(nn,tmp(sa1),1,tmp(sa0),1)
          !sa1 = s22
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
        end if
c        call Dneall_f_GramSchmidt(ms,psi2,npack1)
        notgram = .false.
  630   continue

*       :::::::::::::::::  correction due to the constraint  :::::::::::::::::
        if (notgram) then

!$OMP MASTER
      fmfcount = fmfcount+1
!$OMP END MASTER
          call eDneall_fmf_Multiply(mb,ispin,ne,
     >                              psi1,nida+nidb,
     >                              tmp(sa1), dte,
     >                              psi2,1.0d0)
        end if

!$OMP SINGLE
        call eDneall_mm_Expand(mb,ne,tmp(sa1),lmbda)
!$OMP END SINGLE

  640 continue

c*:::::::::::::::::  correction due to the constraint  :::::::::::::::::
      call nwpw_timing_end(3)

      call stop_collection()
      return
      end

      subroutine eDneall_ffm_zeroMatrix(matrix,m,n)!,treset)
      implicit none
      integer m,n
      real*8 matrix(m*n)
      integer k,j

#if 0
!$OMP SINGLE      
      matrix=0
!$OMP END SINGLE
#else
!$OMP DO schedule(static)
      do k=1,m*n
        matrix(k) = 0.0
      end do
!$OMP END DO
#endif


      end




      subroutine eDneall_ffm_sym_Multiply_reduce_concur(A1,A2,
     >                         nida,nidb,ne,hml,thrhml,
     >           shift,shift2,n,ftid,nwthr,reduce_lock)!,treset,treduce)
      USE omp_lib
      implicit none
      complex*16 A1(*),A2(*)
      integer nida,nidb,ispin,ne(2)
      real*8 hml(*)
      real*8 thrhml(*)

      real*8 ratio
      integer ftid,nwthr,nthrc,nthrr,tidc,tidr
      integer n,shift,shift2,bm,offsetm
      integer nk,bk,offsetk,minchunk,nchunkc,nchunkr
      integer itid,tid,nthr,chunkc,ne1,chunkr
      integer innernthr

      INTEGER(kind=omp_nest_lock_kind) reduce_lock
*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads,Parallel_maxthreads
      external Parallel_threadid,Parallel_nthreads,Parallel_maxthreads
!      double precision treset,treduce

      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()
      nk = 2*nidb

      if( (tid .ge. ftid) .and. (tid .lt. ftid+nwthr)) then 
        !ratio = REAL(nk) / REAL(n)
        !nthrr = max(1,INT(sqrt( REAL(nwthr)/ratio)))
        !nthrc = nwthr / nthrr
        !nwthr = nthrc*nthrr
        nthrr = 1
        nthrc = nwthr / nthrr
        nwthr = nthrc*nthrr

!        bk = max(min(nk,32),floor(REAL(nk)/REAL(nthrc)))
        bk = max(min(nk,1),floor(REAL(nk)/REAL(nthrc)))
        nthrc = floor(REAL(nk) / REAL(bk))
        nwthr = nthrc*nthrr
        
        if(tid .lt. ftid+nwthr) then
          tidc = mod((tid-ftid),nthrc)
          tidr = (tid-ftid) / nthrc
        

          nchunkc = nthrc
          chunkc=tidc

          !compute offsets in A and B
          bk = floor(REAL(nk)/REAL(nchunkc))
          offsetk = (chunkc)*bk
          if(chunkc==nchunkc-1) bk = nk-offsetk
          
          ne1 = ne(1)

          nchunkr = nthrr
          chunkr = tidr
                


          bm = floor(REAL(n)/REAL(nchunkr))
          offsetm = (chunkr)*bm
          if(chunkr==nchunkr-1) bm = n-offsetm

             call epack_ccm_sym_dot_reduce_concur(nida,nidb,n,
     >                               A1(shift),
     >                               A2(shift),
     >                               hml(shift2), 
     >                               thrhml((tid)*ne1*ne1*8+ shift2),
     >                      nk,bk,offsetk,bm,offsetm,reduce_lock)
        end if
      end if
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





      subroutine epack_ccm_sym_dot_reduce_concur(nida,nidb,n,A,B,matrix,
     >                   thrmatrix,nk,bk,offsetk,bm,offsetm,reduce_lock)
      USE omp_lib
      implicit none
      integer    nida,nidb,n
      real*8 A(*)
      real*8 B(*)
      real*8     matrix(n,n)
      real*8     thrmatrix(n,n)

*     **** local variables ****
      integer j,k,nk
      integer np,npack,npack2
      integer tid
      integer offsetk,bk,bm,bkc,offsetm
      integer offsetThread
      integer bkk,ibkk,kk

      INTEGER(kind=omp_nest_lock_kind) reduce_lock


*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads



      tid  = Parallel_threadid()

      !we dont want the gemm to be too big
      !bkk = min(2**floor(log(REAL(100000-n*n)/REAL(2*n))/log(2.0d0)),bk)
      !bkk = min(2*n,bk)
      bkk = bk
      !write(*,*) bkk, "vs",bk

      npack  = (nida+nidb)
      npack2 = 2*npack

      do kk=0,bk-1,bkk
        ibkk = min(bkk,bk-kk)
        !write(*,*) kk, ibkk, bk

        if(kk.eq.0)then
        call DGEMM('T','N',bm,n,ibkk,
     >               2.0d0,
     >               A(1+ 2*nida + offsetk +kk + offsetm*npack2),npack2,
     >               B(1+ 2*nida + offsetk +kk ),npack2,
     >               0.0d0,
     >               thrmatrix(1+offsetm,1), n)
        else
        call DGEMM('T','N',bm,n,ibkk,
     >               2.0d0,
     >               A(1+ 2*nida + offsetk +kk + offsetm*npack2),npack2,
     >               B(1+ 2*nida + offsetk +kk ),npack2,
     >               1.0d0,
     >               thrmatrix(1+offsetm,1), n)
        endif
      end do

      if(tid.eq.0) then
      call DGEMM('T','N',bm,n,2*nida,
     >             1.0d0,
     >             A(1 + offsetm*npack2),npack2,
     >             B,npack2,
     >             1.0d0,
     >             thrmatrix(1+offsetm,1), n)
      end if

      call nwpw_timing_start_thr(47)
      if(offsetm .eq. 0 .and. bm .eq. n) then
        call omp_set_nest_lock(reduce_lock)
!!$OMP CRITICAL
        call daxpy(n*n,(1.0d0),thrmatrix,1,matrix,1)
!!$OMP END CRITICAL
        call omp_unset_nest_lock(reduce_lock)
      else
        call omp_set_nest_lock(reduce_lock)
!!$OMP CRITICAL
        do k=1,n
!$OMP SIMD
          do j=1+offsetm,offsetm+bm
            matrix(j,k) = matrix(j,k) + thrmatrix(j,k) 
          end do
!$OMP END SIMD
        end do
!!$OMP END CRITICAL
        call omp_unset_nest_lock(reduce_lock)
      end if
      call nwpw_timing_end_thr(47)

#ifdef OMPTASK
!!$OMP END TASK
#endif

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

!!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s22,ne)
!!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
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
!!$OMP END PARALLEL

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

!!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s21,ne)
!!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
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
!!$OMP END PARALLEL
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

!!$OMP PARALLEL firstprivate(ms1,ms2,ishift2,dte) shared(s11,ne)
!!$OMP& default(shared) private(k,j,indx,indxt,ms,shift2)
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
!!$OMP END PARALLEL
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





      subroutine eDneall_mmm_Multiply_concur(mb,ispin,ne,A,B,alpha,C,
     >                                  beta,tmp,ms,shift2,n,ftid,nwthr)
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
      integer ftid,nwthr
*     **** matrix_blocking common block ****
      integer mblock(2),nblock(2),algorithm(2)
      common /matrix_blocking/ mblock,nblock,algorithm

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads
      integer  Parallel_index_1dblock
      external Parallel_index_1dblock
      external dlafill

      call Parallel_taskid(taskid)
      call Parallel_np(np)
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

      if( (tid .ge. ftid) .and. (tid .lt. ftid+nwthr)) then 
        tid = tid - ftid
        

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
            pindx = tid + taskid*nwthr
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
               call DGEMM('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                C(ishiftC), n)
            !*** threads and cpus ****
            else
               call dlacpy('G',(mend-mstart),(nend-nstart),
     >                     C(ishiftC),n,tmp(ishiftC),n)
               call DGEMM('N','N',mend-mstart,nend-nstart,n,
     >                alpha,
     >                A(ishiftA), n,
     >                B(ishiftB), n,
     >                beta,
     >                tmp(ishiftC), n)
            end if
          end if

!      if (mb.eq.0) then
!         if ((algorithm(1).lt.1).and.(algorithm(2).lt.1)) then
!            call Parallel_Brdcst_values(MASTER,
!     >                   ne(1)*ne(1)+ne(2)*ne(2),C)
!
!         else if (algorithm(1).lt.1) then
!            call Parallel_Brdcst_values(MASTER,
!     >                   ne(1)*ne(1),C)
!            call Parallel_Vector_SumAll2(
!     >                  ne(2)*ne(2),tmp(ne(1)*ne(1)+1),C(ne(1)*ne(1)+1))
!
!         else if (algorithm(2).lt.1) then
!            call Parallel_Vector_SumAll2(
!     >                   ne(1)*ne(1),tmp,C)
!            call Parallel_Brdcst_values(MASTER,
!     >                   ne(2)*ne(2),C(ne(1)*ne(1)+1))
!
!         else 
!            call Parallel_Vector_SumAll2(
!     >                   ne(1)*ne(1)+ne(2)*ne(2),tmp,C)
!         end if
!      else
!         if (algorithm(mb).lt.1) then
!            call Parallel_Brdcst_values(MASTER,ne(mb)*ne(mb),C)
!         else
!            call Parallel_Vector_SumAll2(ne(mb)*ne(mb),tmp,C)
!         end if
!      end if

      end if
      return
      end





      subroutine eDneall_mmm_Multiply_concur_red(mb,ispin,ne,A,B,alpha,
     >                                  C,beta,tmp,ms,shift2,ftid,nwthr)
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
      integer ftid,nwthr
*     **** matrix_blocking common block ****
      integer mblock(2),nblock(2),algorithm(2)
      common /matrix_blocking/ mblock,nblock,algorithm

*     **** external functions ****
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads
      integer  Parallel_index_1dblock
      external Parallel_index_1dblock
      external dlafill

      call Parallel_taskid(taskid)
      call Parallel_np(np)
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()

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

        
         !if npack / nthr >> ne divide along npack
        !if () then
        !else
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
         !end if
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
