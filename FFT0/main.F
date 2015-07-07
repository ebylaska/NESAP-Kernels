
#define USE_NEWFFT_FORWARD y
#define USE_NEWFFT_BACKWARD y

cccc#define USE_OLDFFT_FORWARD y
cccc#define USE_OLDFFT_BACKWARD y

      program main 

#include "fft.fh"

      parameter (nx2=nx+2,nx21=nx/2+1) 
      parameter (ny21=ny/2+1,nz21=nz/2+1) 
      parameter (nss0=12,nss=6) ! # of dislocation slip systems 
      parameter (pi=3.14159265359) 

      logical oprint
      integer np,taskid,MASTER
      parameter (MASTER=0)


      real*8     y1(n2ft3d_max)
      real*8     y2(n2ft3d_max)
      complex*16 y1k(nfft3d_max)


      real*8   nwpw_timing
      external nwpw_timing

      call nwpw_timing_init()
      call nwpw_timing_start(1)

      call Parallel_Init()
      call Parallel_np(np)
      call Parallel_taskid(taskid)


      call D3dB_Init(1,nx,ny,nz,gridmapping)
      call D3dB_nfft3d(1,nfft3d)
      call D3dB_n2ft3d(1,n2ft3d)
c      nfft3d = (nx/2+1)*ny*nz
c      n2ft3d = (nx+2)*ny*nz
c      taskid = 0
c      np = 1
      Time_step = 1.0d-6
      oprint = (taskid.eq.MASTER)

      if (oprint) then
         write(*,*)
         write(*,*) "taskid               = ",taskid
         write(*,*) "number of processors = ",np
         write(*,*) "nx,ny,nz=",nx,ny,nz
         write(*,*) "nfft3d=",nfft3d,(nx/2+1)*ny*nz
         write(*,*) "n2ft3d=",n2ft3d,(nx+2)*ny*nz
         write(*,*)
      end if

c******** Checking FFTs - should be deleted *******
      do k=1,n2ft3d
        y1(k) = k*0.5
      end do
      call D3dB_r_Zero_Ends(1,y1)
      call forward(y1,y1k,nx,ny,nz)
      call backward(y1k,y2,nx,ny,nz)
      call D3dB_r_Zero_Ends(1,y2)
      c110 = 0.0d0
      do k=1,n2ft3d
         c110 = c110 + dabs(y2(k)-y1(k))
      end do
      write(*,*) "fft error=",c110
      call D3dB_end(1)
      call Parallel_Finalize()
      stop
c******** Checking FFTs - should be deleted *******

       end 


 
#ifdef USE_OLDFFT_FORWARD
      subroutine forward(rin,c,n1,n2,n3)
      implicit none
      integer n1,n2,n3,i,j,k
      logical ifirst
      save ifirst,planfw
      real*8 rin(n1+2,n2,n3)
      real*8 r(n1,n2,n3)
      complex*16 c(n1/2+1,n2,n3)
      real*8 scale
      integer*8  planfw
      data ifirst/.true./
      call nwpw_timing_start(10)

      do k=1,n3
         do j=1,n2
            do i=1,n1
              r(i,j,k) = rin(i,j,k)
            end do
         end do
      end do
      if (ifirst) then
         call dfftw_plan_dft_r2c_3d(planfw,n1,n2,n3,r,c,64)
         ifirst=.false.
      end if
      call dfftw_execute_dft_r2c(planfw,r,c) 
      call nwpw_timing_end(10)
      return 
      end 
#endif
#ifdef USE_NEWFFT_FORWARD
      subroutine forward(rin,c,n1,n2,n3)
      implicit none
      integer n1,n2,n3
      real*8 rin(*)
      complex*16 c(*)
      integer n2ft3d

      call nwpw_timing_start(10)
      call D3dB_n2ft3d(1,n2ft3d)
      call dcopy(n2ft3d,rin,1,c,1)
      call D3dB_rc_fft3f(1,c)
      call nwpw_timing_end(10)
      return 
      end 
#endif


cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#ifdef USE_OLDFFT_BACKWARD
      subroutine backward(c,rout,n1,n2,n3)
      implicit none
      integer n1,n2,n3,i,j,k
      logical ifirst
      save ifirst,planbw
      real*8 r(n1,n2,n3),scale
      real*8 rout(n1+2,n2,n3)
      complex*16 c(n1/2+1,n2,n3),tempk(n1/2+1,n2,n3)
      integer*8  planbw
      data ifirst/.true./

      call nwpw_timing_start(11)
       do i=1,n1/2+1
         do j=1,n2
          do k=1,n3
            tempk(i,j,k)=c(i,j,k)
          enddo
         enddo
        enddo
   
      if (ifirst) then
       call dfftw_plan_dft_c2r_3d(planbw,n1,n2,n3,c,r,64)
       ifirst=.false.
       end if
      scale = 1.d0/(n1*n2*n3) 
       call dfftw_execute_dft_c2r(planbw,c,r)  
       do k=1,n3
           do j=1,n2
              do i=1,n1
                 rout(i,j,k) = scale*r(i,j,k) 
              end do
              rout(n1+1,j,k) = 0.0d0
              rout(n1+2,j,k) = 0.0d0
           end do
       end do

       do i=1,n1/2+1
         do j=1,n2
          do k=1,n3
            c(i,j,k)=tempk(i,j,k)
          enddo
         enddo
        enddo

       call nwpw_timing_end(11)

       return 
       end 
#endif
#ifdef USE_NEWFFT_BACKWARD

      subroutine backward(c,rout,n1,n2,n3)
      implicit none
      integer n1,n2,n3
      real*8 scale
      real*8 rout(*)
      complex*16 c(*)
      integer n2ft3d

      call nwpw_timing_start(11)
      scale = 1.d0/(n1*n2*n3) 
      call D3dB_n2ft3d(1,n2ft3d)
      call dcopy(n2ft3d,c,1,rout,1)
      call D3dB_cr_fft3b(1,rout)
      call dscal(n2ft3d,scale,rout,1)
      call nwpw_timing_end(11)
      return
      end
#endif
   

      subroutine nwpw_timing_init()
      implicit none

      real*8 nwpw_tim1(50),nwpw_tim2(50)
      real*8 times(50)
      common / nwpw_timing_block / times
      common / nwpw_timing2_block / nwpw_tim1,nwpw_tim2
      integer i
      do i=1,50
         times(i) = 0.0d0
      end do
      return
      end
      subroutine nwpw_timing_start(counter)
      implicit none
      integer counter

      real*8 nwpw_tim1(50),nwpw_tim2(50)
      real*8 times(50)
      common / nwpw_timing_block / times
      common / nwpw_timing2_block / nwpw_tim1,nwpw_tim2

      call current_second(nwpw_tim1(counter))
      return
      end
      subroutine nwpw_timing_end(counter)
      implicit none
      integer counter

      real*8 nwpw_tim1(50),nwpw_tim2(50)
      real*8 times(50)
      common / nwpw_timing_block / times
      common / nwpw_timing2_block / nwpw_tim1,nwpw_tim2

      call current_second(nwpw_tim2(counter))

      times(counter) = times(counter)
     >               + (nwpw_tim2(counter)-nwpw_tim1(counter))
      return
      end
      real*8 function nwpw_timing(counter)
      implicit none
      integer counter

      real*8 nwpw_tim1(50),nwpw_tim2(50)
      real*8 times(50)
      common / nwpw_timing_block / times
      common / nwpw_timing2_block / nwpw_tim1,nwpw_tim2

      nwpw_timing = times(counter)
      return
      end
      SUBROUTINE current_second(T)
      real*8 T
      real*4 dummy,etime
      real*4 s(2)
      real   t2(2)

      dummy = etime(s)
      T = dble(s(1))

      RETURN
      END

