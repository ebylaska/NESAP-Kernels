
      subroutine lmbda_test_r(npack1,ne)
#ifdef INTEL
      USE IFPORT
#endif
      implicit none

*     **** arguments ****
      integer npack1, ne(2)

*     **** variables ****
      integer MASTER,taskid,np
      parameter (MASTER=0)
      logical oprint

*     **** psi variables ****
c      integer npack1_max,nemax
c      parameter (npack1_max=5000,nemax=100)

      integer nida,nidb
      complex*16 psi1(npack1 *(ne(1)+ne(2)))
      complex*16 psi2(npack1 *(ne(1)+ne(2)))
      complex*16 dpsi1(npack1*(ne(1)+ne(2)))
      real*8 lmbda(ne(1)*ne(1)*2),tmp(ne(1)*ne(1)*8)

      integer ispin,n1(2),n2(2),ms
      integer i,j,k,n,indxj,indxk,shift,mpierr,provided,ierr
      integer tid,nthr,mthr
      integer number_tests
      real*8 w,ww,dte

*     **** external functions ****
      integer  Parallel_maxthreads,Parallel_index_1dblock
      external Parallel_maxthreads,Parallel_index_1dblock
      integer  Parallel_threadid,Parallel_nthreads
      external Parallel_threadid,Parallel_nthreads

      !**** testing parameters ****
      ispin = 1
      !ne(1) = 100   !* ne(1)+ne(2) <=nemax
      !ne(2) = 0 
      !npack1 = 5000  !* npack1 <= npack1_max
      dte    = 1.0d-2 !* maximum component change in psi
      number_tests = 5
      !**** testing parameters ****


      n1(1) = 1
      n2(1) = ne(1)
      n1(2) = ne(1)+1
      n2(2) = ne(1)+ne(2)

      !call Parallel_Init()

      call nwpw_timing_init()

      call Parallel_np(np)
      call Parallel_taskid(taskid)
      oprint = (taskid.eq.MASTER)

      shift = ne(1)*npack1
      nida = 0
      if (taskid.eq.MASTER) nida = 1
      nidb = npack1-nida


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

      !*** dpsi1 = error ***
      ww = 0.0d0
      do k=1,(ne(1)+ne(2))
         indxk = 1 + (k-1)*npack1
         call epack_cc_dot(nida,nidb,dpsi1(indxk),dpsi1(indxk),w)
         if (dabs(w).gt.dabs(ww)) ww = w
      end do
      ww = 1.0d0/ww
      call dscal(2*npack1*(ne(1)+ne(2)),ww,dpsi1,1)
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
         write(*,*) "ispin                =",ispin
         write(*,*) "ne(1)                =",ne(1)
         write(*,*) "ne(2)                =",ne(2)
         write(*,*) "npack1               =",npack1
         write(*,*)
      end if

      do i=1,number_tests
!         call dcopy(2*npack1*(ne(1)+ne(2)),psi1,1,psi2,1)
!         call daxpy(2*npack1*(ne(1)+ne(2)),dte,dpsi1,1,psi2,1)

!$OMP PARALLEL private(n,k,tid,nthr) 
        tid  = Parallel_threadid()
        nthr = Parallel_nthreads()
         do n=tid,(ne(1)+ne(2)-1),nthr
            do k=1,npack1
               psi2(k+n*npack1)=psi1(k+n*npack1)+dte*dpsi1(k+n*npack1)
            end do
         end do
!$OMP END PARALLEL


         call psi_lmbda(ispin,ne,ne(1)+ne(2),nida,nidb,
     >                  psi1,psi2,dte,lmbda,tmp,ierr)
      end do


      call nwpw_timing_print_final(oprint,number_tests)

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


      program lmbda_test1
      integer npack1, ne(2),np,rank

*     **** get ne and npack from std input ****

      call Parallel_Init()
      call Parallel_np(np)
      call Parallel_taskid(rank)

      if(rank .eq. 0) then
          read(*,*) npack1, ne(1), ne(2)
          npack1 = npack1 / np
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

