      program main 
      implicit none

*     **** The parameters for the calculation are contained fft.fh **** 
*     **** Edit the file and recompile                             ****
#include "fft.fh"

*     **** local variables ****
      integer np,taskid,MASTER
      parameter (MASTER=0)

      logical oprint
      integer it,k,n2ft3d,nfft3d
      real*8  error,cpu1,cpu0
      character*26 dd

      real*8  y1(n2ft3d_max)
      real*8  y2(n2ft3d_max)
      complex*16 y1k(nfft3d_max)


*     **** external functions ****
      real*8   nwpw_timing
      external nwpw_timing


      call nwpw_timing_init()
      call nwpw_timing_start(1)

      call Parallel_Init()
      call Parallel_np(np)
      call Parallel_taskid(taskid)
      oprint = (taskid.eq.MASTER)

      if (oprint) then
         write(*,*)
         write(*,100)
         write(*,101)
         write(*,102)
         write(*,103)
         write(*,104)
         write(*,105)
         write(*,106)
         write(*,107)
         write(*,108)
         write(*,109)
         write(*,110)
         call util_date(dd)
         write(*,*) "       >>> job started at    ",dd," <<<"
         write(*,*)
      end if

      if (oprint) write(*,*) "Initializing D3dB"
      call D3dB_Init(1,nx,ny,nz,gridmapping)
      call D3dB_nfft3d(1,nfft3d)
      call D3dB_n2ft3d(1,n2ft3d)

      if (oprint) then
         write(*,*) 
         write(*,*) "==  Input Data  =="
         write(*,*) 
         write(*,200) taskid
         write(*,201) np
         write(*,*)
         write(*,202) nx,ny,nz
         write(*,203) (nx/2+1)*ny*nz,nfft3d
         write(*,204) (nx+2)*ny*nz,n2ft3d
         write(*,*)
         write(*,205) norbitals
         write(*,206) 2*norbitals+2+7
         write(*,207) niter
         write(*,*)
      end if

      do k=1,n2ft3d
        !y1(k) = dsin(k*9.5d0) + 40.0d0*dcos(6.23424242d0*k)
        y1(k) =  10.987654321d0
      end do
      call D3dB_r_Zero_Ends(1,y1)
     

      if (oprint) then
         call util_date(dd)
         write(*,*) 
         write(*,*) "==  Running the Same Number of 3d FFTS as in a",
     >             " CPMD Simulation  =="
         write(*,*) 
         write(*,*) 
         write(*,300) 
         write(*,301) 
         write(*,*) "       >>> iteration started ",dd," <<<"
      end if
      call nwpw_timing_start(2)
      if (oprint) call current_second(cpu0)

      do it=1,niter
         do k=1,norbitals+1+4
            call fft_forward(nx,ny,nz,y1,y1k)
         end do
         do k=1,norbitals+1+3
            call fft_backward(nx,ny,nz,y1k,y2)
         end do
         call current_second(cpu1)
         if (oprint) then
            call current_second(cpu1)
            write(*,302) it,cpu1-cpu0
         end if
      end do

      if (oprint) then
         call util_date(dd)
         write(*,*) "       >>> iteration ended   ",dd," <<<"
         write(*,*)
      end if
      call nwpw_timing_end(2)

      error = 0.0d0
      do k=1,n2ft3d
         error = error + dabs(y2(k)-y1(k))
      end do
      call D3dB_SumAll(error)
      if (oprint) then
         write(*,*) 
         write(*,*) "==  Summary of Results =="
         write(*,*) 
         write(*,*) "fft error=",error
      end if
      call D3dB_end(1)
      call Parallel_Finalize()

      call nwpw_timing_print_final(oprint,1)
      if (oprint) then
         call util_date(dd)
         write(*,*)
         write(*,*) "       >>> job ended at      ",dd," <<<"
         write(*,*)
      end if

      stop

 100  format(8x,"****************************************************")
 101  format(8x,"*                                                  *")
 102  format(8x,"*     Simplest Parallel 3d FFT Test Calculation    *")
 103  format(8x,"*                                                  *")
 104  format(8x,"*       FFT0-fftpack version 0.0  - 07/03/15       *")
 105  format(8x,"*                                                  *")
 106  format(8x,"*  This code was developed by Eric J. Bylaska for  *")
 107  format(8x,"*  NERSC NESAP Dungeon session with Intel on July  *")
 108  format(8x,"*  8-10, 2015 in Hillsboro, OR                     *")
 109  format(8x,"*                                                  *")
 110  format(8x,"****************************************************")

 200  format(1x,"taskid               = ",I6)
 201  format(1x,"number of processors = ",I6)
 202  format(1x,"FFT nx,ny,nz =",3I5)
 203  format(1x,"nfft3d       =",I10," (",I10," per processor)")
 204  format(1x,"n2ft3d       =",I10," (",I10," per processor)")
 205  format(1x,"number of orbitals                =",I6)
 206  format(1x,"number of FFTs per CPMD iteration =",I6)
 207  format(1x,"number of CPMD iterations         =",I6)
 300  format(8x,"iteration",29x,"current time")
 301  format(8x,"--------------------------------------------------")
 302  format(8x,I5,26x,F11.3,' seconds') 
      end 



      subroutine fft_forward(n1,n2,n3,myrin,myc)
      implicit none
      integer n1,n2,n3
      double precision     myrin(*)
      double complex       myc(*)
      integer n2ft3d,i

      call D3dB_n2ft3d(1,n2ft3d)
      call dcopy(n2ft3d,myrin,1,myc,1)
      call D3dB_rc_fft3f(1,myc)
      return 
      end 


cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine fft_backward(n1,n2,n3,c,rout)
      implicit none
      integer n1,n2,n3
      real*8 rout(*)
      complex*16 c(*)

      real*8 scale
      integer n2ft3d

      scale = 1.0d0/dble(n1*n2*n3) 
      call D3dB_n2ft3d(1,n2ft3d)
      call dcopy(n2ft3d,c,1,rout,1)
      call D3dB_cr_fft3b(1,rout)
      call dscal(n2ft3d,scale,rout,1)
      return
      end
   

