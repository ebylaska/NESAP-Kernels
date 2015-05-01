
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
      integer npack1_max,nemax
      parameter (npack1_max=2500,nemax=10)

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
      ne(1) = 10   !* ne(1)+ne(2) <=nemax
      ne(2) = 0 
      npack1 = 2500  !* npack1 <= npack1_max
      dte    = 1.0d-2 !* maximum component change in psi
      number_tests = 50
      !**** testing parameters ****




      n1(1) = 1
      n2(1) = ne(1)
      n1(2) = ne(1)+1
      n2(2) = ne(1)+ne(2)

      call Parallel_Init()

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
      call nwpw_timing_init()
!$OMP PARALLEL private(i,n,k,tid)
      tid  = Parallel_threadid()
      nthr = Parallel_nthreads()
      do i=1,number_tests

         do n=tid,(ne(1)+ne(2)-1),nthr
            do k=1,npack1
               psi2(k+n*npack1)=psi1(k+n*npack1)+dte*dpsi1(k+n*npack1)
            end do
         end do
         !call dcopy(2*npack1*(ne(1)+ne(2)),psi1,1,psi2,1)
         !call daxpy(2*npack1*(ne(1)+ne(2)),dte,dpsi1,1,psi2,1)
         call psi_lmbda(ispin,ne,ne(1)+ne(2),nida,nidb,
     >                  psi1,psi2,dte,lmbda,tmp,ierr)
      end do
!$OMP END PARALLEL
      call nwpw_timing_print_final(oprint,number_tests)

      !if (oprint) write(*,*) "<psi2|psi2> = "
      !call check_norms(oprint,ispin,ne,nida,nidb,psi2)

      call Parallel_Finalize()
      
 
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
      integer npack1, ne(2)

*     **** get ne and npack from std input ****
      
      read(*,*) npack1, ne(1), ne(2)
      write(*,*) npack1, ne(1), ne(2)
  
!      ne(1) = 10   !* ne(1)+ne(2) <=nemax
!      ne(2) = 0 
!      npack1 = 10000  !* npack1 <= npack1_max

      call lmbda_test_r(npack1,ne)

      end program lmbda_test1
