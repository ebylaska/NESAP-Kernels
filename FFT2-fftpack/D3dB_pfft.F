*
* $Id: D3dB_pfft.F 26682 2015-01-09 23:57:32Z mjacquelin $
*

*     ***********************************************************
*     *                                                         *
*     *                    D3dB_pfft library                    *
*     *                 (MPI implemenation)                     *
*     *                                                         *
*     *   Author - Eric Bylaska                                 *
*     *   date   - 3/23/96                                      *
*     *                                                         *
*     ***********************************************************

*     ***********************************
*     *                                 *
*     *         D3dB_pfft_init          *
*     *                                 *
*     ***********************************

      subroutine D3dB_pfft_init(lattice_unitg,lattice_ggcut)
      implicit none
      real*8 lattice_unitg(3,3),lattice_ggcut(2)

#include "D3dB.fh"

*     **** local variables ****
      integer taskid,MASTER
      parameter (MASTER=0)

      double precision eps
      parameter (eps=1.0d-12)

      logical value,yrow,zrow,yzslab
      integer nxh,nyh,nzh,q,p
      integer k1,k2,k3,ii,nbb
      integer i,j,k
      integer index,index2
      double precision ggcut,g1,g2,g3,gg
      logical zero_arow2((nx/2+1)*nz)
      logical zero_arow3((nx/2+1)*ny)


      call Parallel2d_taskid_i(taskid)
      nxh = nx1(1)/2
      nyh = ny1(1)/2
      nzh = nz1(1)/2


c     ****************************
c     **** slab decomposition ****
c     ****************************
      if (mapping.eq.1) then
         do nbb=0,1

             ggcut = lattice_ggcut(nbb+1)


*           **** find zero_row3 - (i,j,*) rows that are zero ****
            do q=1,(nxh+1)*nq(1)
               zero_row3(q,nbb) =.true.
            end do
            do q=1,(nxh+1)*ny1(1)
               zero_arow3(q) =.true.
            end do


            do k2 = -nyh+1, nyh-1
            do k1 = 0,nxh-1
               i=k1
               j=k2
               if (i.lt.0) i= i+ nx1(1)
               if (j.lt.0) j= j+ ny1(1)
               i=i+1
               j=j+1
               zrow = .true.
               do k3 = -nzh+1, nzh-1
                  g1 = k1*lattice_unitg(1,1)
     >               + k3*lattice_unitg(1,2)
     >               + k2*lattice_unitg(1,3)
                  g2 = k1*lattice_unitg(2,1)
     >               + k3*lattice_unitg(2,2)
     >               + k2*lattice_unitg(2,3)
                  g3 = k1*lattice_unitg(3,1)
     >               + k3*lattice_unitg(3,2)
     >               + k2*lattice_unitg(3,3)
                  gg = g1*g1 + g2*g2 + g3*g3
                  gg= gg-ggcut
                  if (gg.lt.-eps) zrow = .false.
               end do
               if (.not.zrow) then
                  zero_arow3(i+(nxh+1)*(j-1)) =.false.
                  call D3dB_ktoqp(1,j,q,p)
                  if (p.eq.taskid) then
                    index2 = i + (nxh+1)*(q-1)
                    zero_row3(index2,nbb) =.false.
                  end if
               end if
            end do
            end do

            call D3dB_c_ptranspose_jk_init(nbb,zero_arow3)

*           **** find zero_slab23 - (i,*,*) slabs that are zero ****
            do k1 = 1,nxh+1
               zero_slab23(k1,nbb) =.true.
            end do

            do k1=0,nxh-1
               i=k1
               if (i.lt.0) i=i+nx1(1)
               i=i+1
               yzslab = .true.
               do k3 = -nzh+1, nzh-1
               do k2 = -nyh+1, nyh-1
                  g1 = k1*lattice_unitg(1,1)
     >               + k2*lattice_unitg(1,2)
     >               + k3*lattice_unitg(1,3)
                  g2 = k1*lattice_unitg(2,1)
     >               + k2*lattice_unitg(2,2)
     >               + k3*lattice_unitg(2,3)
                  g3 = k1*lattice_unitg(3,1)
     >               + k2*lattice_unitg(3,2)
     >               + k3*lattice_unitg(3,3)
                  gg = g1*g1 + g2*g2 + g3*g3
                  gg= gg-ggcut
                  if (gg.lt.-eps) yzslab = .false.
               end do
               end do
               if (.not.yzslab) then
                  zero_slab23(i,nbb) =.false.
               end if
            end do

*           **** find zero_row2 - (i,*,k) rows that are zero after fft of (i,j,*) ****
            do k3 = 1,nz1(1)
            do k1 = 1,nxh+1
               call D3dB_ktoqp(1,k3,q,p)
               if (p.eq.taskid) then
                  index2 = k1 + (nxh+1)*(q-1)
                  zero_row2(index2,nbb) = zero_slab23(k1,nbb)
               end if
            end do
            end do

         end do !*nbb*


c     *****************************************
c     **** mapping == 2 - 2d decomposition ****
c     *****************************************
      else

         do nbb=0,1

            ggcut = lattice_ggcut(nbb+1)

*           **** find zero_row3 - (i,j,*) rows that are zero ****
            do q = 1,nq3(1)
               zero_row3(q,nbb) =.true.
            end do
            do q = 1,(nxh+1)*ny1(1)
               zero_arow3(q) =.true.
            end do

            do k2 = -nyh+1, nyh-1
            do k1 = 0,nxh-1
               i=k1
               j=k2
               if (i.lt.0) i=i+nx1(1)
               if (j.lt.0) j=j+ny1(1)
               i=i+1
               j=j+1
               zrow = .true.
               do k3 = -nzh+1, nzh-1
                  g1 = k1*lattice_unitg(1,1)
     >               + k2*lattice_unitg(1,2)
     >               + k3*lattice_unitg(1,3)
                  g2 = k1*lattice_unitg(2,1)
     >               + k2*lattice_unitg(2,2)
     >               + k3*lattice_unitg(2,3)
                  g3 = k1*lattice_unitg(3,1)
     >               + k2*lattice_unitg(3,2)
     >               + k3*lattice_unitg(3,3)
                  gg = g1*g1 + g2*g2 + g3*g3
                  gg= gg-ggcut
                  if (gg.lt.-eps) zrow = .false.
               end do
               if (.not.zrow) then
                  zero_arow3(i+(nxh+1)*(j-1)) =.false.
                  q = q_map3(i+(j-1)*(nx1(1)/2+1),1)
                  p = p_map3(i+(j-1)*(nx1(1)/2+1),1)
                  if (p.eq.taskid) then
                    zero_row3(q,nbb) =.false.
                  end if
               end if
            end do
            end do

*           **** find zero_slab23 - (i,*,*) slabs that are zero ****
            do k1 = 1,nxh+1
               zero_slab23(k1,nbb) =.true.
            end do

            do k1 = 0,nxh-1
               i=k1
               if (i .lt. 0) i = i + nx1(1)
               i=i+1
               yzslab = .true.
               do k3 = -nzh+1, nzh-1
               do k2 = -nyh+1, nyh-1
                  g1 = k1*lattice_unitg(1,1)
     >               + k2*lattice_unitg(1,2)
     >               + k3*lattice_unitg(1,3)
                  g2 = k1*lattice_unitg(2,1)
     >               + k2*lattice_unitg(2,2)
     >               + k3*lattice_unitg(2,3)
                  g3 = k1*lattice_unitg(3,1)
     >               + k2*lattice_unitg(3,2)
     >               + k3*lattice_unitg(3,3)
                  gg = g1*g1 + g2*g2 + g3*g3
                  gg= gg-ggcut
                  if (gg.lt.-eps) yzslab = .false.
               end do
               end do
               if (.not.yzslab) then
                  zero_slab23(i,nbb) = .false.
               end if
            end do

*           **** find zero_row2 - (i,*,k) rows that are zero after fft of (i,j,*) ****
            do k = 1,nz1(1)
            do i = 1,nxh+1
               q = q_map2(k+(i-1)*(nz1(1)),1)
               p = p_map2(k+(i-1)*(nz1(1)),1)

               zero_arow2(i+(nxh+1)*(k-1)) = zero_slab23(i,nbb)
               if (p.eq.taskid) then
                  zero_row2(q,nbb) = zero_slab23(i,nbb)
               end if
            end do
            end do

            call D3dB_c_ptranspose_ijk_init(nbb,zero_arow2,zero_arow3)

         end do !*nbb*

      end if

      return
      end











*     ***********************************
*     *					*
*     *	        D3dB_cr_pfft3b		*
*     *					*
*     ***********************************

      subroutine D3dB_cr_pfft3b(nb,nbb,A,nfft3d0)

*****************************************************
*                                                   *
*      This routine performs the operation of       *
*      a three dimensional complex to complex       *
*      inverse fft                                  *
*           A(nx,ny(nb),nz(nb)) <- FFT3^(-1)[A(kx,ky,kz)]   * 
*                                                   *
*      Entry - 					    *
*              A: a column distribuded 3d block     *
*              tmp: tempory work space must be at   *
*                    least the size of (complex)    *
*                    (nfft*nfft + 1) + 10*nfft      * 
*                                                   *
*       Exit - A is transformed and the imaginary   *
*              part of A is set to zero             *
*       uses - D3dB_c_transpose_jk, dcopy           *
*                                                   *
*****************************************************

      implicit none
      integer nb,nbb,nfft3d0
      complex*16  A(nfft3d0)


#include "D3dB.fh"

*     *** local variables ***
      integer i,j,k,q,indx
      integer nxh,nxhy,nxhz,indx0,indx1,mythr

*     **** external functions ****
      integer  Parallel_threadid
      external Parallel_threadid

      
      call nwpw_timing_start(1)

*     ***** allocate temporary space ****
      !call D3dB_nfft3d(nb,nfft3d)

       nxh = (nx1(nb)/2+1)
       nxhz = nxh*nz1(nb)
       nxhy = nxh*ny1(nb)
      
      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ********************************************
*     ***         Do a transpose of A          ***
*     ***      A(kx,kz,ky) <- A(kx,ky,kz)      ***
*     ********************************************
cccc  XXXX shouldn't be called XXXX
c      call D3dB_c_transpose_jk(nb,A,dtmp2,dtmp3)

*     *************************************************
*     ***     do fft along kz dimension             ***
*     ***   A(kx,nz(nb),ky) <- fft1d^(-1)[A(kx,kz,ky)]  ***
*     *************************************************
      call nwpw_timing_start(3)
      indx0 = 0
      do q=1,nq(nb)
         do i=1,nxh

            indx  = i + indx0
            indx1 = indx
            do k=1,nz1(nb)
               dtmp2(k) = A(indx)
               indx = indx + nxh
            end do
            call dcfftb(nz1(nb),dtmp2,tmpz(1,nb,1))
            do k=1,nz1(nb)
               A(indx1) = dtmp2(k)
               indx1 = indx1 + nxh
            end do

         end do
         indx0 = indx0 + nxhz
      end do
      call nwpw_timing_end(3)


*     ********************************************
*     ***         Do a transpose of A          ***
*     ***      A(kx,ky,nz(nb)) <- A(kx,nz(nb),ky)      ***
*     ********************************************
      call nwpw_timing_start(4)
      call D3dB_c_transpose_jk(nb,A,dtmp2,dtmp3)
      call nwpw_timing_end(4)

*     *************************************************
*     ***     do fft along ky dimension             ***
*     ***   A(kx,ny(nb),nz(nb)) <- fft1d^(-1)[A(kx,ky,nz(nb))]  ***
*     *************************************************
      call nwpw_timing_start(3)
      indx0 = 0
      do q=1,nq(nb)
         do i=1,(nx1(nb)/2+1)

            indx  = i + indx0
            indx1 = indx
            do j=1,ny1(nb)
               dtmp2(j) = A(indx)
               indx = indx + nxh
            end do
            call dcfftb(ny1(nb),dtmp2,tmpy(1,nb,1))
            do j=1,ny1(nb)
               A(indx1) = dtmp2(j)
               indx1 = indx1 + nxh
            end do

         end do
         indx0 = indx0 + nxhy
      end do


*     *************************************************
*     ***     do fft along kx dimension             ***
*     ***   A(nx(nb),ny(nb),nz(nb)) <- fft1d^(-1)[A(kx,ny(nb),nz(nb))]  ***
*     *************************************************
      call cshift1_fftb(nx1(nb),ny1(nb),nq(nb),1,A)
      indx = 1
      do q=1,nq(nb)
      do j=1,ny1(nb)
         !indx = 1 + (j-1)*(nx(nb)/2+1) + (q-1)*(nx(nb)/2+1)*ny(nb)
         call drfftb(nx1(nb),A(indx),tmpx(1,nb,1))
         indx = indx + nxh
      end do
      end do
      call zeroend_fftb(nx1(nb),ny1(nb),nq(nb),1,A)
      call nwpw_timing_end(3)


      !*************************
      !**** hilbert mapping ****
      !*************************
      else


*     *************************************************
*     ***     do fft along kz dimension             ***
*     ***   A(nz(nb),kx,ky) <- fft1d^(-1)[A(kz,kx,ky)]  ***
*     *********************t****************************
      call nwpw_timing_start(3)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq3(nb)
        call dcfftb(nz1(nb),A(1+(i-1)*nz1(nb)),tmpz(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL

      call nwpw_timing_end(3)

      call nwpw_timing_start(4)
      call D3dB_c_transpose_ijk(nb,3,A,dtmp2,dtmp3,nfft3d(nb))
      call nwpw_timing_end(4)

*     *************************************************
*     ***     do fft along ky dimension             ***
*     ***   A(ny(nb),nz(nb),kx) <- fft1d^(-1)[A(ky,nz(nb),kx)]  ***
*     *************************************************
      call nwpw_timing_start(3)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq2(nb)
        call dcfftb(ny1(nb),A(1+(i-1)*ny1(nb)),tmpy(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL
      call nwpw_timing_end(3)

      call nwpw_timing_start(4)
      call D3dB_c_transpose_ijk(nb,4,A,dtmp2,dtmp3,nfft3d(nb))
      call nwpw_timing_end(4)

*     *************************************************
*     ***     do fft along kx dimension             ***
*     ***   A(nx(nb),ny(nb),nz(nb)) <- fft1d^(-1)[A(kx,ny(nb),nz(nb))]  ***
*     *************************************************
      call nwpw_timing_start(3)
      call cshift1_fftb(nx1(nb),nq1(nb),1,1,A)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq1(nb)
        call drfftb(nx1(nb),A(1+(i-1)*nxh),tmpx(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL
      call zeroend_fftb(nx1(nb),nq1(nb),1,1,A)
      call nwpw_timing_end(3)


      end if


      call nwpw_timing_end(1)
      return
      end


*     ***********************************
*     *					*
*     *	        D3dB_rc_pfft3f		*
*     *					*
*     ***********************************

      subroutine D3dB_rc_pfft3f(nb,nbb,A,nfft3d0)

*****************************************************
*                                                   *
*      This routine performs the operation of       *
*      a three dimensional complex to complex fft   *
*           A(kx,ky,kz) <- FFT3[A(nx(nb),ny(nb),nz(nb))]        * 
*                                                   *
*      Entry - 					    *
*              A: a column distribuded 3d block     *
*              tmp: tempory work space must be at   *
*                    least the size of (complex)    *
*                    (nfft*nfft + 1) + 10*nfft      * 
*                                                   *
*       Exit - A is transformed                     *
*                                                   *
*       uses - transpose1 subroutine                *
*                                                   *
*****************************************************

      implicit none
      integer nb,nbb,nfft3d0
      complex*16  A(nfft3d0)

#include "D3dB.fh"


*     *** local variables ***
      integer i,j,k,q,indx,indx1
      integer nxh,nxhy,nxhz,mythr

*     **** external functions ****
      integer  Parallel_threadid
      external Parallel_threadid

      call nwpw_timing_start(1)


*     ***** allocate temporary space ****
      nxh = (nx1(nb)/2+1)
      nxhz = nxh*nz1(nb)
      nxhy = nxh*ny1(nb)


      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ********************************************
*     ***     do fft along nx(nb) dimension        ***
*     ***   A(kx,ny(nb),nz(nb)) <- fft1d[A(nx(nb),ny(nb),nz(nb))]  ***
*     ********************************************
      call nwpw_timing_start(3)
      indx = 1
      do q=1,nq(nb)
      do j=1,ny1(nb)
         call drfftf(nx1(nb),A(indx),tmpx)
         indx = indx + nxh
      end do
      end do
      call cshift_fftf(nx1(nb),ny1(nb),nq(nb),1,A)


*     ********************************************
*     ***     do fft along ny(nb) dimension        ***
*     ***   A(kx,ky,nz(nb)) <- fft1d[A(kx,ny(nb),nz(nb))]  ***
*     ********************************************
      do i=1,nxh
      indx = i
      indx1= i
         do q=1,nq(nb)
            do j=1,ny1(nb)
               dtmp2(j) = A(indx)
               indx = indx + nxh
            end do

            call dcfftf(ny1(nb),dtmp2,tmpy(1,nb,1))

            do j=1,ny1(nb)
               A(indx1) = dtmp2(j)
               indx1 = indx1 + nxh
            end do
         end do

      end do
      call nwpw_timing_end(3)


*     ********************************************
*     ***         Do a transpose of A          ***
*     ***      A(ky,nz(nb),ky) <- A(kx,ky,nz(nb))      ***
*     ********************************************
      call nwpw_timing_start(4)
      call D3dB_c_transpose_jk(nb,A,dtmp2,dtmp3)
      call nwpw_timing_end(4)


*     ********************************************
*     ***     do fft along nz(nb) dimension        ***
*     ***   A(kx,kz,ky) <- fft1d[A(kx,nz(nb),ky)]  ***
*     ********************************************
      call nwpw_timing_start(3)
      do i=1,nxh
         indx  = i
         indx1 = i
         do q=1,nq(nb)
            do k=1,nz1(nb)
               dtmp2(k) = A(indx)
               indx = indx + nxh
            end do

            call dcfftf(nz1(nb),dtmp2,tmpz(1,nb,1))

            do k=1,nz1(nb)
               A(indx1) = dtmp2(k)
               indx1 = indx1 + nxh
            end do
         end do
      end do
      call nwpw_timing_end(3)


*     ********************************************
*     ***         Do a transpose of A          ***
*     ***      A(kx,ky,kz) <- A(kx,kz,ky)      ***
*     ********************************************
ccc XXXX shouldn't  be called XXXX
c      call D3dB_c_transpose_jk(nb,A,dtmp2,dtmp3)

      !*************************
      !**** hilbert mapping ****
      !*************************
      else

*     ********************************************
*     ***     do fft along nx(nb) dimension        ***
*     ***   A(kx,ny(nb),nz(nb)) <- fft1d[A(nx(nb),ny(nb),nz(nb))]  ***
*     ********************************************
      call nwpw_timing_start(3)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq1(nb)
        call drfftf(nx1(nb),A(1+(i-1)*nxh),tmpx(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL

      call cshift_fftf(nx1(nb),nq1(nb),1,1,A)
      call nwpw_timing_end(3)


      call nwpw_timing_start(4)
      call D3dB_c_transpose_ijk(nb,1,A,dtmp2,dtmp3,nfft3d(nb))
      call nwpw_timing_end(4)

*     ********************************************
*     ***     do fft along ny(nb) dimension        ***
*     ***   A(ky,nz(nb),kx) <- fft1d[A(ny(nb),nz(nb),kx)]  ***
*     ********************************************
      call nwpw_timing_start(3)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq2(nb)
        call dcfftf(ny1(nb),A(1+(i-1)*ny1(nb)),tmpy(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL
      call nwpw_timing_end(3)

      call nwpw_timing_start(4)
      call D3dB_c_transpose_ijk(nb,2,A,dtmp2,dtmp3,nfft3d(nb))
      call nwpw_timing_end(4)

*     ********************************************
*     ***     do fft along nz(nb) dimension        ***
*     ***   A(kz,kx,ky) <- fft1d[A(nz(nb),kx,ky)]  ***
*     ********************************************
      call nwpw_timing_start(3)
!$OMP PARALLEL private(i,mythr)
      mythr = Parallel_threadid()
!$OMP DO schedule(dynamic,100)
      do i=1,nq3(nb)
        call dcfftf(nz1(nb),A(1+(i-1)*nz1(nb)),tmpz(1,nb,mythr+1))
      end do
!$OMP END DO
!$OMP END PARALLEL
      call nwpw_timing_end(3)

      end if


      call nwpw_timing_end(1)
      return
      end



*     ***********************************
*     *					*
*     *	   D3dB_c_ptranspose_jk_init	*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_jk_init(nbb,zero_arow3)
      implicit none
      integer nbb
      logical zero_arow3(*)

#include "D3dB.fh"


c     integer iq_to_i1((NFFT1/2+1)*NFFT2*NSLABS)
c     integer iq_to_i2((NFFT1/2+1)*NFFT2*NSLABS)
c     integer i1_start(NFFT3+1)
c     integer i2_start(NFFT3+1)
      integer iq_to_i1(nfft3d_max,0:1)
      integer iq_to_i2(nfft3d_max,0:1)
      integer i1_start(nz+1,0:1)
      integer i2_start(nz+1,0:1)
      common / ptrans_blk / iq_to_i1,iq_to_i2,i1_start,i2_start


*     **** local variables ****
      integer proc_to,proc_from
      integer pto,qto,np,taskid
      integer pfrom,qfrom
      integer phere,qhere
      integer index1,index2,itmp
      integer i,j,k,it
      logical value


      call Parallel2d_taskid_i(taskid)
      call Parallel2d_np_i(np)

      index1 = 1 
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         i1_start(it+1,nbb) = index1
         i2_start(it+1,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)

*           **** packing scheme **** 
            call D3dB_ktoqp(1,k,qhere,phere)
            call D3dB_ktoqp(1,j,qto,pto)

            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               do i=1,(nx1(1)/2+1)
                  itmp = i + (j-1)*(nx1(1)/2+1) 
     >                     + (qhere-1)*(nx1(1)/2+1)*ny1(1)
c                 iq_to_i1(itmp) = index1
                  iq_to_i1(itmp,nbb) = index1
                  index1 = index1 + 1
               end do
            end if
             
*           **** unpacking scheme ****
            call D3dB_ktoqp(1,j,qhere,phere)
            call D3dB_ktoqp(1,k,qfrom,pfrom)
            if ((phere.eq.taskid).and.(pfrom.eq.proc_from)) then
               do i=1,(nx1(1)/2+1)
                  itmp = i + (k-1)*(nx1(1)/2+1) 
     >                     + (qhere-1)*(nx1(1)/2+1)*ny1(1)
c                 iq_to_i2(itmp) = index2
                  iq_to_i2(itmp,nbb) = index2
                  index2 = index2 + 1
               end do
            end if
         end do
         end do
      end do
      i1_start(np+1,nbb) = index1
      i2_start(np+1,nbb) = index2

      return
      end


*     ***********************************
*     *					*
*     *	   D3dB_c_ptranspose2_jk	*
*     *					*
*     ***********************************

      subroutine D3dB_c_ptranspose2_jk(nbb,A,tmp1,tmp2)

*****************************************************
*                                                   *
*      This routine performs the operation          *
*               A(i,k,j) <- A(i,j,k)                * 
*                                                   *
*      np = the number of worker nodes              *
*      proc#=0...(np-1)
*                                                   *
*       this transpose uses more buffer space       *
*       then transpose2                             *
*****************************************************
      implicit none
      integer nbb
      complex*16  A(*)
      complex*16  tmp1(*),tmp2(*)

#include "mpif.h"
#include "D3dB.fh"

*     **** indexing variables ****
      integer iq_to_i1(nfft3d_max,0:1)
      integer iq_to_i2(nfft3d_max,0:1)
      integer i1_start(nz+1,0:1)
      integer i2_start(nz+1,0:1)
      common / ptrans_blk / iq_to_i1,iq_to_i2,i1_start,i2_start



*     **** local variables ***
      logical value
      integer i
      integer proc_to,proc_from,it
      integer msgtype,source
      integer msglen,type
      integer request(np_max+1),reqcnt
      integer taskid,np
      integer mpierr

*     **** external functions ****
      integer  Parallel2d_convert_taskid_i
      external Parallel2d_convert_taskid_i
      
      call Parallel2d_taskid_i(taskid)
      call Parallel2d_np_i(np)


*     **** pack A(i) array ****
      do i=1,nfft3d(1) 
         tmp1(iq_to_i1(i,nbb)) = A(i)
      end do

*     **** it = 0, transpose data on same thread ****  
      msglen = i2_start(2,nbb) - i2_start(1,nbb)

      do i=1,msglen
         tmp2(i2_start(1,nbb)+i-1) = tmp1(i1_start(1,nbb)+i-1)
      end do
         

*     **** receive packed array data ****
      reqcnt = 0
      do it = 1,np-1

*        **** synchronous receive of tmp ****
         proc_from = mod((taskid)-it+np, np) 

         msgtype = 1
         source=proc_from
         type   = 1
         msglen = i2_start(it+2,nbb) - i2_start(it+1,nbb)
         
         if (msglen.gt.0) then
            reqcnt = reqcnt + 1
            call MPI_IRECV(tmp2(i2_start(it+1,nbb)),
     >                    msglen,MPI_DOUBLE_COMPLEX,
     >                    Parallel2d_convert_taskid_i(source),
     >                    msgtype,MPI_COMM_WORLD,
     >                    request(reqcnt),mpierr) 
         end if

      end do

*     **** send packed array to other processors ****
      do it = 1,np-1

*        **** synchronous send of tmp ****
         proc_to   = mod((taskid)+it, np) 
         msglen    = i1_start(it+2,nbb) - i1_start(it+1,nbb)
         msgtype   = 1
      
         if (msglen.gt.0) then
            call MPI_SEND(tmp1(i1_start(it+1,nbb)),
     >                     msglen,MPI_DOUBLE_COMPLEX,
     >                     Parallel2d_convert_taskid_i(proc_to),
     >                     msgtype,MPI_COMM_WORLD,mpierr)
         end if

      end do

   
*     **** wait for completion of mp_send, also do a sync ****
      if (np.gt.1) call Parallel_mpiWaitAll(reqcnt,request)


*     **** unpack A(i) array ****
      do i=1,nfft3d(1) !***(nx(nb)/2+1)*ny1(nb)*nq(nb)
         A(i) = tmp2(iq_to_i2(i,nbb))
      end do

      return
      end




*     ***********************************
*     *					*
*     *	   D3dB_c_ptranspose_ijk	*
*     *					*
*     ***********************************

      subroutine D3dB_c_ptranspose_ijk(nbb,op,A,tmp1,tmp2,nfft3d0)

*****************************************************
*                                                   *
*      This routine performs the operation          *
*               A(i,k,j) <- A(i,j,k)                * 
*                                                   *
*      np = the number of worker nodes              *
*      proc#=0...(np-1)
*                                                   *
*       this transpose uses more buffer space       *
*       then transpose2                             *
*****************************************************
      implicit none
      integer nbb,op,nfft3d0
      complex*16  A(nfft3d0)
      complex*16  tmp1(nfft3d0),tmp2(nfft3d0)

#include "D3dB.fh"
#include "mpif.h"


*     **** indexing variables ****
      integer h_iq_to_i1(nfft3d_max,6,0:1)
      integer h_iq_to_i2(nfft3d_max,6,0:1)
      integer h_i1_start(np_max+1,6,0:1)
      integer h_i2_start(np_max+1,6,0:1)
      common / ptrans_blk_ijk / h_iq_to_i1,
     >                         h_iq_to_i2,
     >                         h_i1_start,
     >                         h_i2_start



*     **** local variables ***
      logical value
      integer i,nnfft3d
      integer proc_to,proc_from,it
      integer msgtype,source
      integer msglen,type
      integer request(np_max),reqcnt
      integer taskid,np
      integer mpierr

*     **** external functions ****
      integer  Parallel2d_convert_taskid_i
      external Parallel2d_convert_taskid_i

      
      call Parallel2d_taskid_i(taskid)
      call Parallel2d_np_i(np)


*     **** pack A(i) array ****
      if ((op.eq.1).or.(op.eq.5)) nnfft3d = (nx1(1)/2+1)*nq1(1)
      if ((op.eq.2).or.(op.eq.4)) nnfft3d = (ny1(1))    *nq2(1)
      if ((op.eq.3).or.(op.eq.6)) nnfft3d = (nz1(1))    *nq3(1)

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,nnfft3d
         tmp1(h_iq_to_i1(i,op,nbb)) = A(i)
      end do
!$OMP END PARALLEL DO


*     **** it = 0, transpose data on same thread ****  
      msglen = h_i2_start(2,op,nbb)
     >       - h_i2_start(1,op,nbb)
      call dcopy(2*msglen,tmp1(h_i1_start(1,op,nbb)),1,
     >                    tmp2(h_i2_start(1,op,nbb)),1)

cc!$OMP PARALLEL DO private(i) schedule(dynamic,100) default(shared)
cc      do i=1,msglen
cc         tmp2(h_i2_start(1,op,nb)+i)= tmp1(h_i1_start(1,op,nb)+i)
cc      end do
cccc!$OMP END PARALLEL DO
         

*     **** receive packed array data ****
      reqcnt = 0
      do it = 1,np-1

*        **** synchronous receive of tmp ****
         proc_from = mod((taskid)-it+np, np) 

         msgtype = 1
         source=proc_from
         type   = 1
         msglen = h_i2_start(it+2,op,nbb) - h_i2_start(it+1,op,nbb)
         

         if (msglen.gt.0) then
            reqcnt = reqcnt + 1
            call MPI_IRECV(tmp2(h_i2_start(it+1,op,nbb)),
     >                    msglen,MPI_DOUBLE_COMPLEX,
     >                    Parallel2d_convert_taskid_i(source),
     >                    msgtype,MPI_COMM_WORLD,
     >                    request(reqcnt),mpierr) 
         end if

      end do

*     **** send packed array to other processors ****
      do it = 1,np-1

*        **** synchronous send of tmp ****
         proc_to   = mod((taskid)+it, np) 
         msglen    = h_i1_start(it+2,op,nbb) - h_i1_start(it+1,op,nbb)
         msgtype   = 1
      
         if (msglen.gt.0) then
            call MPI_SEND(tmp1(h_i1_start(it+1,op,nbb)),
     >                     msglen,MPI_DOUBLE_COMPLEX,
     >                     Parallel2d_convert_taskid_i(proc_to),
     >                     msgtype,MPI_COMM_WORLD,mpierr)
         end if

      end do

   
*     **** wait for completion of mp_send, also do a sync ****
      if (np.gt.1) call Parallel_mpiWaitAll(reqcnt,request)


*     **** unpack A(i) array ****
      if ((op.eq.4).or.(op.eq.6)) nnfft3d = (nx1(1)/2+1)*nq1(1)
      if ((op.eq.1).or.(op.eq.3)) nnfft3d = (ny1(1))    *nq2(1)
      if ((op.eq.2).or.(op.eq.5)) nnfft3d = (nz1(1))    *nq3(1)

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,nnfft3d
         A(i) = tmp2(h_iq_to_i2(i,op,nbb))
      end do 
!$OMP END PARALLEL DO

      return
      end




*     ***********************************
*     *					*
*     *	   D3dB_c_ptranspose_ijk_init	*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_ijk_init(nbb,zero_arow2,zero_arow3)
      implicit none
      integer nbb
      logical zero_arow2(*),zero_arow3(*)

#include "D3dB.fh"


*     *** hilbert tranpose data structure ****
      integer h_iq_to_i1(nfft3d_max,6,0:1)
      integer h_iq_to_i2(nfft3d_max,6,0:1)
      integer h_i1_start(np_max+1,6,0:1)
      integer h_i2_start(np_max+1,6,0:1)
      common / ptrans_blk_ijk / h_iq_to_i1,
     >                         h_iq_to_i2,
     >                         h_i1_start,
     >                         h_i2_start

*     **** local variables ****
      logical value
      integer proc_to,proc_from
      integer pto,qto,np,taskid
      integer phere,qhere
      integer index1,index2,itmp
      integer i,j,k,it


      call Parallel2d_taskid_i(taskid)
      call Parallel2d_np_i(np)


*     ********************************************************
*     **** map1to2 mapping - done - tranpose operation #1 ****
*     ********************************************************

*     **** allocate trans_blk_ijk common block ****
      index1 = 1 
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,1,nbb) = index1
         h_i2_start(it+1,1,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme **** 
            phere = p_map1((j-1)+(k-1)*ny1(1)+1,1)
            qhere = q_map1((j-1)+(k-1)*ny1(1)+1,1)

            pto   = p_map2((k-1)+(i-1)*nz1(1)+1,1)
            qto   = q_map2((k-1)+(i-1)*nz1(1)+1,1)
            
           
            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = i + (qhere-1)*(nx1(1)/2+1)
               h_iq_to_i1(itmp,1,nbb) = index1
               index1 = index1 + 1
            end if
             
*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = j + (qto-1)*ny1(1)
               h_iq_to_i2(itmp,1,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,1,nbb) = index1
      h_i2_start(np+1,1,nbb) = index2





*     *********************************************************
*     **** map2to3 mapping - done - transpose operation #2 ****
*     *********************************************************

*     **** allocate trans_blk_ijk common block ****
      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,2,nbb) = index1
         h_i2_start(it+1,2,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme ****
            phere = p_map2((k-1)+(i-1)*nz1(1)+1,1)
            qhere = q_map2((k-1)+(i-1)*nz1(1)+1,1)

            pto   = p_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)
            qto   = q_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)

            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = j + (qhere-1)*ny1(1)
               h_iq_to_i1(itmp,2,nbb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = k + (qto-1)*nz1(1)
               h_iq_to_i2(itmp,2,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,2,nbb) = index1
      h_i2_start(np+1,2,nbb) = index2






*     ********************************************************
*     **** map3to2 mapping - done - tranpose operation #3 ****
*     ********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,3,nbb) = index1
         h_i2_start(it+1,3,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme ****
            phere = p_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)
            qhere = q_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)

            pto   = p_map2((k-1)+(i-1)*nz1(1)+1,1)
            qto   = q_map2((k-1)+(i-1)*nz1(1)+1,1)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = k + (qhere-1)*nz1(1)
               h_iq_to_i1(itmp,3,nbb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = j + (qto-1)*ny1(1)
               h_iq_to_i2(itmp,3,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,3,nbb) = index1
      h_i2_start(np+1,3,nbb) = index2




*     ********************************************************
*     **** map2to1 mapping - done - tranpose operation #4 ****
*     ********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,4,nbb) = index1
         h_i2_start(it+1,4,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme ****
            phere = p_map2((k-1)+(i-1)*nz1(1)+1,1)
            qhere = q_map2((k-1)+(i-1)*nz1(1)+1,1)

            pto   = p_map1((j-1)+(k-1)*ny1(1)+1,1)
            qto   = q_map1((j-1)+(k-1)*ny1(1)+1,1)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = j + (qhere-1)*ny1(1)
               h_iq_to_i1(itmp,4,nbb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = i + (qto-1)*(nx1(1)/2+1)
               h_iq_to_i2(itmp,4,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,4,nbb) = index1
      h_i2_start(np+1,4,nbb) = index2





*     **********************************************************
*     **** map1to3 mapping  - done - tranpose operation # 5 ****
*     **********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,5,nbb) = index1
         h_i2_start(it+1,5,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme ****
            phere = p_map1((j-1)+(k-1)*ny1(1)+1,1)
            qhere = q_map1((j-1)+(k-1)*ny1(1)+1,1)

            pto   = p_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)
            qto   = q_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = i + (qhere-1)*(nx1(1)/2+1)
               h_iq_to_i1(itmp,5,nbb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = k + (qto-1)*nz1(1)
               h_iq_to_i2(itmp,5,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,5,nbb) = index1
      h_i2_start(np+1,5,nbb) = index2






*     *************************
*     **** map3to1 mapping ****
*     *************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,6,nbb) = index1
         h_i2_start(it+1,6,nbb) = index2

         do k=1,nz1(1)
         do j=1,ny1(1)
         do i=1,(nx1(1)/2+1)

*           **** packing scheme ****
            phere = p_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)
            qhere = q_map3((i-1)+(j-1)*(nx1(1)/2+1)+1,1)

            pto   = p_map1((j-1)+(k-1)*ny1(1)+1,1)
            qto   = q_map1((j-1)+(k-1)*ny1(1)+1,1)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = k + (qhere-1)*nz1(1)
               h_iq_to_i1(itmp,6,nbb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = i + (qto-1)*(nx1(1)/2+1)
               h_iq_to_i2(itmp,6,nbb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,6,nbb) = index1
      h_i2_start(np+1,6,nbb) = index2

      return
      end


*     ***********************************
*     *                                 *
*     *        generate_map_indexes     *
*     *                                 *
*     ***********************************

      subroutine generate_map_indexes(taskid,np,ny,nz,p_map,q_map,nq)
      implicit none
      integer taskid,np
      integer ny,nz
      integer p_map(*)
      integer q_map(*)
      integer nq



      !**** local variables ****
      logical value
      integer i,j,k,q,p
      integer nq1,nq2,rmdr1
      integer indx_proc(ny*nz),indx_q(ny*nz)


      nq1   = (ny*nz)/np
      rmdr1 = mod(ny*nz,np)
      nq2 = nq1
      if (rmdr1.gt.0) nq2 = nq2+1
      nq = 0
      p=0
      q=1
      do i=1,(ny*nz)
        indx_proc(i) = p
        indx_q(i)    = q

        if (taskid.eq.p) nq = nq + 1
        q = q+1
        if (q.gt.nq2) then
           q = 1
           p = p + 1
           p = mod(p,np)
           if (p.ge.rmdr1) nq2 = nq1
        end if
      end do


      do k=1,nz
      do j=1,ny
        i = p_map(j+(k-1)*ny)+1
        p = indx_proc(i)
        q = indx_q(i)
        p_map(j+(k-1)*ny) = p
        q_map(j+(k-1)*ny) = q
      end do
      end do

      return
      end

*     ***********************************
*     *                                 *
*     *        expand_hilbert2d         *
*     *                                 *
*     ***********************************

      subroutine expand_hilbert2d(np,ny,nz, 
     >                            pmap_in, qmap_in,
     >                            pmap_out,qmap_out)
      implicit none
      integer np,ny,nz
      integer pmap_in(ny,nz)
      integer qmap_in(ny,nz)
      integer pmap_out(2*ny,2*nz) 
      integer qmap_out(2*ny,2*nz)

      !**** local variables ****
      integer j,k,p,nqp


      !*** set pmap_out ***
      do k=1,nz
      do j=1,ny
         pmap_out(j,k)       = pmap_in(j,k)
         pmap_out(j+ny,k)    = pmap_in(j,k)
         pmap_out(j,k+nz)    = pmap_in(j,k)
         pmap_out(j+ny,k+nz) = pmap_in(j,k)
      end do
      end do


      !*** set qmap_out ***
      do p=0,np-1

         nqp=0
         do k=1,nz
         do j=1,ny
           if (pmap_in(j,k).eq.p) then
              if (qmap_in(j,k).gt.nqp) nqp = qmap_in(j,k)
           end if
         end do
         end do

         do k=1,nz
         do j=1,ny
           if (pmap_in(j,k).eq.p) then
              qmap_out(j,k)       = qmap_in(j,k)
              qmap_out(j+ny,k)    = qmap_in(j,k) + nqp
              qmap_out(j,k+nz)    = qmap_in(j,k) + 2*nqp
              qmap_out(j+ny,k+nz) = qmap_in(j,k) + 3*nqp
           end if
         end do
         end do

      end do

      return
      end
         



*     ***********************************
*     *                                 *
*     *        generate_map_indexes_2   *
*     *                                 *
*     ***********************************

      subroutine generate_map_indexes_2(taskid,np,ny,nz,p_map,q_map,nq)
      implicit none
      integer taskid,np
      integer ny,nz
      integer p_map(*)
      integer q_map(*)
      integer nq


      !**** local variables ****
      logical value
      integer i,j,k,q,p
      integer nq1,nq2,rmdr1
      integer indx_proc(ny*nz),indx_q(ny*nz)

      nq = 0
      p=0
      q=1
      do i=1,(ny*nz)
        indx_proc(i) = p
        indx_q(i)    = q
        if (taskid.eq.p) nq = nq + 1

        p = p + 1
        if (p.ge.np) then
          p = 0
          q = q+1
        end if
      end do


      do k=1,nz
      do j=1,ny
        i = p_map(j+(k-1)*ny)+1
        p = indx_proc(i)
        q = indx_q(i)
        p_map(j+(k-1)*ny) = p
        q_map(j+(k-1)*ny) = q
      end do
      end do

      return
      end


*     ***********************************
*     *                                 *
*     *        generate_map_indexes_3   *
*     *                                 *
*     ***********************************

      subroutine generate_map_indexes_3(taskid,np,ny,nz,p_map,q_map,nq)
      implicit none
      integer taskid,np
      integer ny,nz
      integer p_map(*)
      integer q_map(*)
      integer nq


      !**** local variables ****
      logical value
      integer i,j,k,q,p,jj,kk,p1,q1,p2,q2
      integer nq1,nq2,rmdr1


      nq = 0
      p=0
      q=1
      do k=1,nz
      do j=1,ny
        p_map(j+(k-1)*ny) = p
        q_map(j+(k-1)*ny) = q
        if (taskid.eq.p) nq = nq + 1

        p = p + 1
        if (p.ge.np) then
          p = 0
          q = q+1
        end if
      end do
      end do

      do k=0,(nz/2-1)
        i = k+2
        j = nz-k
        p1 = p_map(1+(i-1)*ny)
        q1 = q_map(1+(i-1)*ny)
        p2 = p_map(1+(j-1)*ny)
        q2 = q_map(1+(j-1)*ny)

        value = .false.
        jj=2
        kk=1
        do while (.not.value)
          p = p_map(jj+(kk-1)*ny)
          q = q_map(jj+(kk-1)*ny)

          value = (p.eq.p1)
          if (value) then
             p_map(jj+(kk-1)*ny) = p2
             q_map(jj+(kk-1)*ny) = q2
             p_map(1+(j-1)*ny)   = p
             q_map(1+(j-1)*ny)   = q
          else
             kk = kk+1
             if (kk.gt.nz) then
               kk=1
               jj=jj+1
             end if
          end if
        end do

      end do

      return
      end


