      program main 
      implicit none

*     **** The parameters for the calculation are contained fft.fh **** 
*     **** Edit the file and recompile                             ****
#include "fft.fh"

*     **** local variables ****
      integer np,taskid,MASTER,nthr
      parameter (MASTER=0)

      logical oprint,oprint2
      integer it,k,n2ft3d,nfft3d,mythr
      real*8  error,cpu0,cpu1,cpu2,cpu3,rc_time,cr_time
      real*8 fftx_flop,ffty_flop,fftz_flop,fft3d_flop,cr_flop,rc_flop
      character*26 dd


      real*8  y1(n2ft3d_max)
      real*8  y2(n2ft3d_max)
      complex*16 y1k(nfft3d_max)


*     **** external functions ****
      integer  Parallel_maxthreads,Parallel_threadid
      external Parallel_maxthreads,Parallel_threadid
      real*8   nwpw_timing
      external nwpw_timing


      call nwpw_timing_init()
      call nwpw_timing_start(1)

      call Parallel_Init()
      call Parallel_np(np)
      call Parallel_taskid(taskid)
      nthr   = Parallel_maxthreads()
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
         write(*,211) nthr
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
      rc_time = 0.0d0
      cr_time = 0.0d0
      call nwpw_timing_start(2)
      if (oprint) call current_second(cpu0)
!$OMP PARALLEL private(k,it,oprint2,mythr) shared(y1,y2,y1k)
!$OMP BARRIER
      mythr = Parallel_threadid()
      oprint2 = oprint.and.(Parallel_threadid().eq.MASTER)

      do it=1,niter
         if (oprint2) call current_second(cpu1)
         do k=1,norbitals+1+4
            call fft_forward(nx,ny,nz,y1,n2ft3d,y1k,nfft3d)
         end do
         if (oprint2) call current_second(cpu2)
         do k=1,norbitals+1+3
            call fft_backward(nx,ny,nz,y1k,nfft3d,y2,n2ft3d)
         end do
         if (oprint2) call current_second(cpu3)
         if (oprint2) then
            rc_time = rc_time + (cpu2-cpu1)
            cr_time = cr_time + (cpu3-cpu2)
            write(*,302) it,cpu3-cpu0
         end if
      end do
!$OMP END PARALLEL

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

      fftx_flop = 2.5d0*nx*dlog(dble(nx))/dlog(2.0d0)
      ffty_flop = 5.0d0*ny*dlog(dble(ny))/dlog(2.0d0)
      fftz_flop = 5.0d0*nz*dlog(dble(nz))/dlog(2.0d0)
      fft3d_flop = fftx_flop * ny*nz
     >           + ffty_flop * (nx/2+1)*nz
     >           + fftz_flop * (nx/2+1)*ny

      cr_flop = (norbitals+1+3)*(fft3d_flop + (nx+2)*ny*nz)
      rc_flop = (norbitals+1+4)*fft3d_flop


      if (oprint) then
         write(*,*) 
         write(*,*) "==  Summary of Results =="
         write(*,*) 
         write(*,*) "fft error=",error
         write(*,*)
         write(*,400) fftx_flop
         write(*,401) ffty_flop
         write(*,402) fftz_flop
         write(*,403) fft3d_flop
         write(*,404) niter*(norbitals+1+4),niter*rc_flop
         write(*,405) niter*(norbitals+1+3),niter*cr_flop
         write(*,406) niter*(norbitals+1+4),rc_time,
     >                rc_time/dble(niter*(norbitals+1+4))
         write(*,407) niter*(norbitals+1+3),cr_time,
     >                cr_time/dble(niter*(norbitals+1+3))
         write(*,408) niter*rc_flop/rc_time,rc_flop/rc_time*1.0d-9
         write(*,409) niter*cr_flop/cr_time,cr_flop/cr_time*1.0d-9
         write(*,*)
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
 211  format(1x,"number of threads    = ",I6)
 300  format(8x,"iteration",29x,"current time")
 301  format(8x,"--------------------------------------------------")
 302  format(8x,I5,26x,F11.3,' seconds') 
 400  format(/1x,"number flops per fftx   = ", E12.6)
 401  format(1x,"number flops per ffty   = ",E12.6)
 402  format(1x,"number flops per fftz   = ",E12.6)
 403  format(1x,"number flops per 3d fft = ", E12.6)
 404  format(/1x,"number flops for ",I5,
     >           " (complex to real) cr_ffts = ",E12.6)
 405  format(1x,"number flops for ", I5,
     >          " (real to complex) rc_ffts = ",E12.6)
 406  format(/1x,"Time for ",I5," rc_fft3d in seconds = ",E12.6,
     >       " (",E12.6," seconds per fft)")
 407  format(1x,"Time for ",I5," cr_fft3d in seconds = ",E12.6,
     >       " (",E12.6," seconds per fft)")
 408  format(/1x,"flop rate for rc_fft3d = ",E12.6,
     >       " flops/second (",E12.6," Gflops)")
 409  format(1x,"flop rate for cr_fft3d = ",E12.6,
     >       " flops/second (",E12.6," Gflops)")

      end 


      subroutine fft_forward(n1,n2,n3,myrin,n2ft3d,myc,nfft3d)
      implicit none
      integer n1,n2,n3,n2ft3d,nfft3d
      double precision     myrin(n2ft3d)
      double complex       myc(nfft3d)

      call D3dB_rr_Copy(1,myrin,myc,n2ft3d)
      call D3dB_rc_fft3f(1,myc,nfft3d)
      return 
      end 


cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine fft_backward(n1,n2,n3,c,nfft3d,rout,n2ft3d)
      implicit none
      integer n1,n2,n3,nfft3d,n2ft3d
      complex*16 c(nfft3d)
      real*8 rout(n2ft3d)

      real*8 scale

      scale = 1.0d0/dble(n1*n2*n3) 
      call D3dB_rr_Copy(1,c,rout,n2ft3d)
      call D3dB_cr_fft3b(1,rout,nfft3d)
      call D3dB_r_SMul1(1,scale,rout,n2ft3d)
      return
      end
   

