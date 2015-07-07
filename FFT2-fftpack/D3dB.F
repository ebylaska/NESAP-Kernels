*     ***********************************************************
*     *								*
*     *   		   D3dB library				*
*     *			(MPI implemenation)			*
*     *								*
*     *   Author - Eric Bylaska					*
*     *   date   - 3/23/96					*
*     *								*
*     ***********************************************************

*	The D3dB (distributed three-dimensional block) library is to 
* be used for handling three kinds of data structures.  The first 
* data structure, denoted by "r", is a double precision array of
* length (nx(nb)+2)*ny(nb)*nz.  The second data structure, denoted by "c", is
* a double complex array of length of (nx(nb)/2+1)*ny(nb)*nz.  The third data

* (nx(nb)/2+1)*ny(nb)*nz.
*
*	The three data structures are distributed across threads, p, in
* the k (i.e. nz(nb)) dimension using a cyclic decomposition.  So that
* a "r" array A is defined as double precision A(nx(nb)+2,ny(nb),nq(nb)) on
* each thread.  
*
*	Where 
*		np = number of threads
*		nq(nb) = ceil(nz(nb)/np).
*		0 <= p < np
*		1 <= q <= nq(nb)
*		1 <= k <= nz(nb)
* 
* 	The mapping of k -> q is defined as:
*	
*		k = ((q-1)*np + p) + 1
*		q = ((k-1) - p)/np + 1
*		p = (k-1) mod np
*
*  Libraries used: mpi, blas, fftpack, and compressed_io
*
*  common blocks used in this library:
*
*       integer nq,nx(NBLOCKS),ny,nz(nb)
*	common	/ D3dB / nq,nx,ny,nz
*
*	integer q_map(NFFT3),p_map(NFFT3),k_map(NFFT3)
* 	common /D3dB_mapping / q_map,p_map,k_map
*
*     integer iq_to_i1((NFFT1/2+1)*NFFT2*NSLABS)
*     integer iq_to_i2((NFFT1/2+1)*NFFT2*NSLABS)
*     integer i1_start(NPROCS+1)
*     integer i2_start(NPROCS+1)
*     common / trans_blk / iq_to_i1,iq_to_i2,i1_start,i2_start

*     **** local variables ****

*     ***********************************
*     *					*
*     *	       Mapping_Init		*	
*     *					*
*     ***********************************

      subroutine Mapping_Init(nb)
      implicit none
      integer nb
     
#include  "D3dB.fh"

      integer k,q,p
*     integer kn
      integer taskid,np
      logical value
      
      call Parallel2d_np_i(np)
      call Parallel2d_taskid_i(taskid)

*     **************************
*     ****** Slab mapping ******
*     **************************
      if (mapping.eq.1) then

*     ****** cyclic ******
       p = 0
       q = 1
       do k=1,nz1(nb)
         q_map(k,nb) = q
         p_map(k,nb) = p
         if (p .eq. taskid) nq(nb) = q

         p        = p+1
         if (p .ge. np) then
            p = 0
            q = q + 1
         end if
       end do

      !*** not used anymore!! ****
      do k=1,nz1(nb)
         if (p_map(k,nb) .eq. taskid) then
            k_map(q_map(k,nb),nb) = k
         end if
      end do

      nfft3d(nb)     = (nx1(nb)/2+1)*ny1(nb)*nq(nb)
      n2ft3d(nb)     = 2*nfft3d(nb)
      nfft3d_map(nb) = nfft3d(nb)
      n2ft3d_map(nb) = n2ft3d(nb)


*     ******************************
*     ****** Hilbert mappings ******
*     ******************************
      else


      !**** double grid map1 defined wrt to single grid         ****
      !**** makes expand and contract routines trivial parallel ****

      if (mapping2d.eq.1) then
         if (nb.eq.1) then
           call hilbert2d_map(ny1(nb),nz1(nb),p_map1(1,nb))
         end if
         call hilbert2d_map(nz1(nb),(nx1(nb)/2+1),p_map2(1,nb))
         call hilbert2d_map((nx1(nb)/2+1),ny1(nb),p_map3(1,nb))
      else
         if (nb.eq.1) then
           call hcurve_map(ny1(nb),nz1(nb),p_map1(1,nb))
         end if
        
         call hcurve_map(nz1(nb),(nx1(nb)/2+1),p_map2(1,nb))
         call hcurve_map((nx1(nb)/2+1),ny1(nb),p_map3(1,nb))
      end if


      !**** double grid map1 defined wrt to single grid         ****
      !**** makes expand and contract routines trivial parallel ****
      if (nb.eq.1) then
      call generate_map_indexes(taskid,np,
     >                          ny1(nb),nz1(nb),
     >                          p_map1(1,nb),
     >                          q_map1(1,nb),nq1(nb))
      else
        nq1(2) = 4*nq1(1)
        call expand_hilbert2d(np,ny1(1),nz1(1),
     >                        p_map1(1,1),q_map1(1,1),
     >                        p_map1(1,2),q_map1(1,2))
      end if
      call generate_map_indexes(taskid,np,
     >                          nz1(nb),nx1(nb)/2+1,
     >                          p_map2(1,nb),
     >                          q_map2(1,nb),nq2(nb))
      call generate_map_indexes(taskid,np,
     >                          nx1(nb)/2+1,ny1(nb),
     >                          p_map3(1,nb),
     >                          q_map3(1,nb),nq3(nb))


      nfft3d(nb) = (nx1(nb)/2+1)*nq1(nb)
      if ((ny1(nb)*nq2(nb)).gt.nfft3d(nb)) nfft3d(nb) = ny1(nb)*nq2(nb)
      if ((nz1(nb)*nq3(nb)).gt.nfft3d(nb)) nfft3d(nb) = nz1(nb)*nq3(nb)
      n2ft3d(nb) = 2*nfft3d(nb)

      nfft3d_map(nb) = nz1(nb)*nq3(nb)
      n2ft3d_map(nb) = (nx1(nb)+2)*nq1(nb)
      
      end if

      return
      end


*     ***********************************
*     *					*
*     *	          D3dB_end   		*	
*     *					*
*     ***********************************
      subroutine D3dB_end(nb)
      implicit none
      integer nb

#include "D3dB.fh"


      logical value
      integer i

      call D3dB_fft_end(nb)

      return
      end
         
*     ***********************************
*     *					*
*     *	          D3dB_qtok   		*	
*     *					*
*     ***********************************

      subroutine D3dB_qtok(nb,q,k)      
      implicit none
      integer nb
      integer q,k

#include "D3dB.fh"

      k = k_map(q,nb)

      return
      end

*     ***********************************
*     *					*
*     *	          D3dB_ktoqp  		*	
*     *					*
*     ***********************************

      subroutine D3dB_ktoqp(nb,k,q,p)      
      implicit none
      integer nb
      integer k,q,p

#include "D3dB.fh"

      q = q_map(k,nb)
      p = p_map(k,nb)
      return
      end


*     ***********************************
*     *					*
*     *	          D3dB_ijktoindexp	*	
*     *					*
*     ***********************************

      subroutine D3dB_ijktoindexp(nb,i,j,k,indx,p)      
      implicit none
      integer nb
      integer i,j,k
      integer indx,p

#include "D3dB.fh"

      integer q

      !**** slab mapping ***
      if (mapping.eq.1) then
      q = q_map(k,nb)
      p = p_map(k,nb)

      indx = i + (j-1)*(nx1(nb)/2+1) + (q-1)*(nx1(nb)/2+1)*ny1(nb)

      !**** hilbert mapping ****
      else
      q = q_map3(i+(j-1)*(nx1(nb)/2+1),nb)
      p = p_map3(i+(j-1)*(nx1(nb)/2+1),nb)
       
      indx = k + (q-1)*nz1(nb)
      end if
      return
      end



*     ***********************************
*     *                                 *
*     *           D3dB_ijktoindex1p     *
*     *                                 *
*     ***********************************

      subroutine D3dB_ijktoindex1p(nb,i,j,k,indx,p)
      implicit none
      integer nb
      integer i,j,k
      integer indx,p

#include "D3dB.fh"

      integer q

      !**** slab mapping ***
      if (mapping.eq.1) then
      q = q_map(j,nb)
      p = p_map(j,nb)

      indx = i + (k-1)*(nx1(nb)/2+1) + (q-1)*(nx1(nb)/2+1)*nz1(nb)

      !**** hilbert mapping ****
      else
      q = q_map2(k+(i-1)*nz1(nb),nb)
      p = p_map2(k+(i-1)*nz1(nb),nb)

      indx = j + (q-1)*ny1(nb)
      end if
      return
      end


*     ***********************************
*     *                                 *
*     *           D3dB_ijktoindex2p     *
*     *                                 *
*     ***********************************

      subroutine D3dB_ijktoindex2p(nb,i,j,k,indx,p)
      implicit none
      integer nb
      integer i,j,k
      integer indx,p

#include "D3dB.fh"

      integer q

      !**** slab mapping ****
      if (mapping.eq.1) then
      q = q_map(j,nb)
      p = p_map(j,nb)

      indx = i + (k-1)*(nx1(nb)+2) + (q-1)*(nx1(nb)+2)*ny1(nb)

      !**** hilbert mapping ****
      else
      q = q_map1(j+(k-1)*ny1(nb),nb)
      p = p_map1(j+(k-1)*ny1(nb),nb)
      
      indx = i + (q-1)*(nx1(nb)+2)
      end if
      return
      end



*     ***********************************
*     *					*
*     *	        D3dB_nfft3d		*
*     *					*
*     ***********************************

      subroutine D3dB_nfft3d(nb,nfft3d_out)
      implicit none
      integer nb
      integer nfft3d_out

#include "D3dB.fh"

      nfft3d_out = nfft3d(nb)
      return
      end

*     ***********************************
*     *					*
*     *	        D3dB_n2ft3d		*
*     *					*
*     ***********************************

      subroutine D3dB_n2ft3d(nb,n2ft3d_out)
      implicit none
      integer nb
      integer n2ft3d_out

#include "D3dB.fh"

      n2ft3d_out = n2ft3d(nb)
      return
      end


*     ***********************************
*     *					*
*     *	        D3dB_nq			*	
*     *					*
*     ***********************************

      subroutine D3dB_nq(nb,nqtmp)
      implicit none
      integer nb
      integer nqtmp

#include "D3dB.fh"

      nqtmp = nq(nb)
      return 
      end

*     ***********************************
*     *					*
*     *	        D3dB_nx			*	
*     *					*
*     ***********************************
     
      subroutine D3dB_nx(nb,nxtmp)
      implicit none
      integer nb
      integer nxtmp

#include "D3dB.fh"

      nxtmp = nx1(nb)
      return
      end

*     ***********************************
*     *					*
*     *	        D3dB_ny			*	
*     *					*
*     ***********************************

      subroutine D3dB_ny(nb,nytmp)
      implicit none
      integer nb
      integer nytmp
      
#include "D3dB.fh"

      nytmp = ny1(nb)
      return
      end

*     ***********************************
*     *					*
*     *	        D3dB_nz			*	
*     *					*
*     ***********************************

      subroutine D3dB_nz(nb,nztmp)
      implicit none
      integer nb
      integer nztmp

#include "D3dB.fh"
      
      nztmp = nz1(nb)
      return
      end


*     ***********************************
*     *					*
*     *	        D3dB_Init		*	
*     *					*
*     ***********************************

      subroutine D3dB_Init(nb,nx_in,ny_in,nz_in,map_in)
      implicit none
      integer nb
      integer nx_in,ny_in,nz_in
      integer map_in

#include "D3dB.fh"

      !**** local variables ****
      integer MASTER
      parameter (MASTER=0)
      integer taskid,np

      call Parallel2d_np_i(np)
      call Parallel_taskid(taskid)


      !**** Make sure ngrid is consistent with mapping ***
      if (map_in.eq.1) then
        if ((np.gt.nz_in).or.(ny_in.ne.nz_in)) then
          if (taskid.eq.MASTER) then
            write(6,*) 'Error: for slab decomposition the',
     >                 ' number of processors must ',
     >                 ' be in the range ( 1 ...ngrid(3)=',
     >                   nz_in,')'
           write(6,*) ' and ngrid(2) == ngrid(3), ',
     >                ' ngrid(2)=',ny_in,
     >                ' ngrid(3)=',nz_in
          end if
          stop
        end if
        if (mod(nx_in,2).ne.0) then
          if (taskid.eq.MASTER) then
           write(6,*)
     >      'Error: ngrid(1) must be even (ngrid(1) mod 2 == 0)'
           write(6,*) 'Error: ngrid(1)=',nx_in
          end if
          stop
        end if
      end if

      if (map_in.ge.2) then
        if (np.gt.(ny_in*nz_in)) then
          if (taskid.eq.MASTER) then
           write(6,*) 'Error: np > MIN(ngrid(2)*ngrid(3),',
     >                ' (ngrid(1)/2+1)*ngrid(2),',
     >                ' (ngrid(1)/2+1)*ngrid(3))'
           write(6,*) 'Error: np > ngrid(2)*ngrid(3)'
           write(6,*) 'Error: for the Hilbert decomposition the',
     >                 ' the number of processors must ',
     >                 ' be in the range ( 1 ...',
     >                   ny_in*nz_in,')'
          end if
          stop
        end if
        if (np.gt.((nx_in/2+1)*ny_in)) then
          if (taskid.eq.MASTER) then
           write(6,*) 'Error: np > MIN(ngrid(2)*ngrid(3),',
     >                ' (ngrid(1)/2+1)*ngrid(2),',
     >                ' (ngrid(1)/2+1)*ngrid(3))'
           write(6,*) 'Error: np > (ngrid(1)/2+1)*ngrid(2)'
           write(6,*) 'Error: for the Hilbert decomposition the',
     >                 ' the number of processors must ',
     >                 ' be in the range ( 1 ...',
     >                   (nx_in/2+1)*ny_in,')'
          end if
          stop
        end if
        if (np.gt.((nx_in/2+1)*nz_in)) then
          if (taskid.eq.MASTER) then
           write(6,*) 'Error: np > MIN(ngrid(2)*ngrid(3),',
     >                ' (ngrid(1)/2+1)*ngrid(2),',
     >                ' (ngrid(1)/2+1)*ngrid(3))'
           write(6,*) 'Error: np > (ngrid(1)/2+1)*ngrid(3)'
           write(6,*) 'Error: for the Hilbert decomposition the', 
     >                 ' the number of processors must ',
     >                 ' be in the range ( 1 ...',
     >                   (nx_in/2+1)*nz_in,')'
          end if
          stop
        end if
        if (mod(nx_in,2).ne.0) then
          if (taskid.eq.MASTER) then
           write(6,*)
     >      'Error: ngrid(1) must be even (ngrid(1) mod 2 == 0)'
           write(6,*) 'Error: ngrid(1)=',nx_in
          end if
          stop
        end if
      end if


*     ***** initialize D3dB common block *****
      nx1(nb)     = nx_in
      ny1(nb)     = ny_in
      nz1(nb)     = nz_in
      mapping    = map_in
      mapping2d  = 1
      if (mapping.eq.3) then
         mapping   = 2
         mapping2d = 2
      end if
         

*     **** do other initializations ****
      call Mapping_Init(nb)
      if (mapping.eq.1) call D3dB_c_transpose_jk_init(nb)
      if (mapping.eq.2) call D3dB_c_transpose_ijk_init(nb)

      call D3dB_fft_init(nb)
      
      return
      end


*     ***********************************
*     *					*
*     *	        D3dB_fft_init		*
*     *					*
*     ***********************************

      subroutine D3dB_fft_init(nb)
      implicit none
      integer nb

#include "D3dB.fh"

      logical value
      integer i,mthr, Parallel_maxthreads
      external Parallel_maxthreads

      mthr = Parallel_maxthreads()

      write(*,*) "maxthread=",mthr
      do i=1,mthr
        !write(*,*) "DEBUG init fft arrays of thread ", i-1
        call drffti(nx1(nb),tmpx(1,nb,i))
        call dcffti(ny1(nb),tmpy(1,nb,i))
        call dcffti(nz1(nb),tmpz(1,nb,i))
      end do


      return
      end

*     ***********************************
*     *                                 *
*     *         D3dB_fft_end            *
*     *                                 *
*     ***********************************

      subroutine D3dB_fft_end(nb)
      implicit none
      integer nb

#include "D3dB.fh"

      logical value

      return
      end



*     ***********************************
*     *					*
*     *	        D3dB_cr_fft3b		*
*     *					*
*     ***********************************

      subroutine D3dB_cr_fft3b(nb,A,nfft3d0)

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
      integer nb,nfft3d0
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



      subroutine cshift1_fftb(nx,ny,nq,ne,A)
      implicit none
      integer nx,ny,nq,ne
      real*8 A(*)

      integer i,j,indx

      !indx = 1
!$OMP PARALLEL DO private(i,j,indx) schedule(dynamic,100)
      do j=1,(ny*nq*ne)
         indx = 1 + (j-1)*(nx+2)
         do i=2,nx
            A(indx+i-1) = A(indx+i)
         end do
         !indx = indx + (nx+2)
      end do
!$OMP END PARALLEL DO

      return
      end 

      subroutine zeroend_fftb(nx,ny,nq,ne,A)
      implicit none
      integer nx,ny,nq,ne
      real*8 A(*)

      integer i,indx
      !indx  = nx+1

!$OMP PARALLEL DO private(i,indx) schedule(dynamic,100)
      do i=1,(ny*nq*ne)
         indx      = nx+1 + (i-1)*(nx+2)
         A(indx)   = 0.0d0
         A(indx+1) = 0.0d0
         !indx      = indx + (nx+2)
      end do
!$OMP END PARALLEL DO

      return
      end 

*     ***********************************
*     *					*
*     *	        D3dB_rc_fft3f		*
*     *					*
*     ***********************************

      subroutine D3dB_rc_fft3f(nb,A,nfft3d0)

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
      integer nb,nfft3d0
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


      subroutine cshift_fftf(nx,ny,nq,ne,A)
      implicit none
      integer nx,ny,nq,ne
      real*8 A(*)

      integer i,j,indx

      !indx = 1

!$OMP PARALLEL DO private(i,j,indx) schedule(dynamic,100)
      do j=1,(ny*nq*ne)
         indx = 1 + (j-1)*(nx+2)
         do i=nx,2,-1
            A(indx+i) = A(indx+i-1)
         end do
         A(indx+1)    = 0.0d0
         A(indx+nx+1) = 0.0d0
         !indx = indx + (nx+2)
      end do
!$OMP END PARALLEL DO

      return
      end 


      subroutine cshift_fftf_ab(nx,ny,nq,ne,A,B)
      implicit none
      integer nx,ny,nq,ne
      real*8 A(*)
      real*8 B(*)

      integer i,j,indx

      indx = 1
      do j=1,(ny*nq*ne)
         do i=nx,2,-1
            B(indx+i) = A(indx+i-1)
         end do
         B(indx+1)    = 0.0d0
         B(indx+nx+1) = 0.0d0
         indx = indx + (nx+2)
      end do

      return
      end






*     ***********************************
*     *					*
*     *	       D3dB_(c,r,t)_SMul 	*	
*     *					*
*     ***********************************

*  This routine performs the operation	C = scale * A
* where scale is a real*8 number.

      subroutine D3dB_c_SMul(nb,scale,A,C)     
      implicit none 
      integer    nb
      real*8     scale
      complex*16 A(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = scale*A(i)
      end do
      return
      end


      subroutine D3dB_r_SMul(nb,scale,A,C)     
      implicit none 
      integer nb
      real*8     scale
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = scale*A(i)
      end do
      return
      end 

      subroutine D3dB_r_SMul1(nb,scale,A,n2ft3d0)
      implicit none
      integer nb,n2ft3d0
      real*8     scale
      real*8 A(n2ft3d0)

#include "D3dB.fh"

      integer i

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,n2ft3d_map(nb)
         A(i) = scale*A(i)
      end do
!$OMP END PARALLEL DO 
      return
      end



      subroutine D3dB_t_SMul(nb,scale,A,C)     
      implicit none 
      integer nb
      real*8 scale
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = scale*A(i)
      end do
      return
      end 

      subroutine D3dB_c_ZMul(nb,scale,A,C)     
      implicit none 
      integer    nb
      complex*16 scale
      complex*16 A(*)
      complex*16 C(*)

#include "D3dB.fh"


      integer i

      do i=1,nfft3d_map(nb)
         C(i) = scale*A(i)
      end do
      return
      end



*     ***********************************
*     *					*
*     *	       D3dB_ct_Sqr	 	*	
*     *					*
*     ***********************************

*  This routine performs the operation	C = A * A

      subroutine D3dB_ct_Sqr(nb,A,C)     
      implicit none 
      integer    nb
      complex*16 A(*)
      real*8     C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = dble(A(i))**2 + dimag(A(i))**2
      end do
      return
      end

*     ***********************************
*     *					*
*     *	       D3dB_rr_Sqr	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_Sqr(nb,A,C)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"


      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = A(i)**2
      end do
      return
      end



*     ***********************************
*     *                                 *
*     *        D3dB_rr_Sqr1             *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Sqr1(nb,A)
      implicit none
      integer nb
      real*8 A(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         A(i) = A(i)**2
      end do
      return
      end


*     ***********************************
*     *					*
*     *	       D3dB_rr_Sqrt	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_Sqrt(nb,A,C)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = dsqrt(A(i))
      end do
      return
      end

*     ***********************************
*     *                                 *
*     *        D3dB_rr_Copy             *       
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Copy(nb,A,C,n2ft3d0)
      implicit none
      integer nb,n2ft3d0
      real*8 A(n2ft3d0)
      real*8 C(n2ft3d0)

#include "D3dB.fh"

      integer i

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,n2ft3d_map(nb)
         C(i) = A(i)
      end do
!$OMP END PARALLEL DO 
      return
      end




*     ***********************************
*     *                                 *
*     *        D3dB_rr_Sqrt1            *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Sqrt1(nb,A)
      implicit none
      integer nb
      real*8 A(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         A(i) = dsqrt(A(i))
      end do
      return
      end


*     ***********************************
*     *					*
*     *	       D3dB_tt_Sqr	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_tt_Sqr(nb,A,C)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = A(i)**2
      end do
      return
      end

             

*     ***********************************
*     *					*
*     *	   D3dB_c_transpose_jk_init	*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_jk_init(nb)
      implicit none
      integer nb

#include "D3dB.fh"


c     integer iq_to_i1((NFFT1/2+1)*NFFT2*NSLABS)
c     integer iq_to_i2((NFFT1/2+1)*NFFT2*NSLABS)
c     integer i1_start(NFFT3+1)
c     integer i2_start(NFFT3+1)
      integer iq_to_i1(nfft3d_max,NBLOCKS)
      integer iq_to_i2(nfft3d_max,NBLOCKS)
      integer i1_start(nz+1,NBLOCKS)
      integer i2_start(nz+1,NBLOCKS)
      common / trans_blk / iq_to_i1,iq_to_i2,i1_start,i2_start


*     **** local variables ****
      integer proc_to,proc_from
      integer pto,qto,np,taskid
      integer pfrom,qfrom
      integer phere,qhere
      integer index1,index2,itmp
      integer i,j,k,it
      logical value


*     **** allocate trans_blk common block ****
      call Parallel2d_taskid_i(taskid)
      call Parallel2d_np_i(np)

      index1 = 1 
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         i1_start(it+1,nb) = index1
         i2_start(it+1,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)

*           **** packing scheme **** 
            call D3dB_ktoqp(nb,k,qhere,phere)
            call D3dB_ktoqp(nb,j,qto,pto)

            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               do i=1,(nx1(nb)/2+1)
                  itmp = i + (j-1)*(nx1(nb)/2+1) 
     >                     + (qhere-1)*(nx1(nb)/2+1)*ny1(nb)
c                 iq_to_i1(itmp) = index1
                  iq_to_i1(itmp,nb) = index1
                  index1 = index1 + 1
               end do
            end if
             
*           **** unpacking scheme ****
            call D3dB_ktoqp(nb,j,qhere,phere)
            call D3dB_ktoqp(nb,k,qfrom,pfrom)
            if ((phere.eq.taskid).and.(pfrom.eq.proc_from)) then
               do i=1,(nx1(nb)/2+1)
                  itmp = i + (k-1)*(nx1(nb)/2+1) 
     >                     + (qhere-1)*(nx1(nb)/2+1)*ny1(nb)
c                 iq_to_i2(itmp) = index2
                  iq_to_i2(itmp,nb) = index2
                  index2 = index2 + 1
               end do
            end if
         end do
         end do
      end do
      i1_start(np+1,nb) = index1
      i2_start(np+1,nb) = index2

      return
      end




*     ***********************************
*     *					*
*     *	         D3dB_cc_dot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_dot(nb,A,B,sumall)     
      implicit none
      integer nb 
      complex*16 A(*)
      complex*16 B(*)
      real*8     sumall


#include "D3dB.fh"

      integer i,j,k,q,index,np,taskid,p
      real*8  sum


      call nwpw_timing_start(2)

      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ***** kx!=0 plane, so double count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
         do i=2,(nx1(nb)/2+1)
            index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >            + (j-1)*(nx1(nb)/2+1) + i
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
         end do
         end do
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
            i=1
            index = (q-1)*(nx1(nb)/2+1)*ny1(nb) + (j-1)*(nx1(nb)/2+1)+1
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
         end do
      end do

      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     ***** kx!=0 plane, so double count *****
      do index=1,nfft3d_map(nb)
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do k=1,nz1(nb)
      do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(1,i,j,k,index,p)
         if (p.eq.taskid) then
         sum = sum - dble(A(index))  * dble(B(index))
     >             - dimag(A(index)) * dimag(B(index))
         end if
      end do
      end do

      end if
      

*     **** add up sums from other nodes ****
      if (np.gt.1) then
         call D3dB_SumAll(sum)
      end if

      call nwpw_timing_end(2)

      sumall = sum
      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_cc_idot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_idot(nb,A,B,sumall)     
      implicit none
      integer nb 
      complex*16 A(*)
      complex*16 B(*)
      real*8     sumall


#include "D3dB.fh"

      integer i,j,k,q,index,np,taskid,p
      real*8  sum


      call nwpw_timing_start(2)

c      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ***** kx!=0 plane, so double count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
         do i=2,(nx1(nb)/2+1)
            index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >            + (j-1)*(nx1(nb)/2+1) + i
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
         end do
         end do
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
            i=1
            index = (q-1)*(nx1(nb)/2+1)*ny1(nb) + (j-1)*(nx1(nb)/2+1)+1
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
         end do
      end do
      

      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     ***** kx!=0 plane, so double count *****
      do index=1,nfft3d_map(nb)
            sum = sum + dble(A(index))  * dble(B(index))
     >                + dimag(A(index)) * dimag(B(index))
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do k=1,nz1(nb)
      do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(1,i,j,k,index,p)
         if (p.eq.taskid) then
         sum = sum - dble(A(index))  * dble(B(index))
     >             - dimag(A(index)) * dimag(B(index))
         end if
      end do
      end do

      end if


*     **** do not add up sums from other nodes ****
       
      call nwpw_timing_end(2)

      sumall = sum
      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_tt_dot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_tt_dot(nb,A,B,sumall)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 B(*)
      real*8 sumall

#include "D3dB.fh"

      integer i,j,k,q,index,np,nxh,taskid,p
      real*8  sum

      nxh=nx1(nb)/2
      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ***** k!=0 plane, so double count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
         do i=2,(nxh+1)
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + i
            sum = sum + A(index)*B(index)
         end do
         end do
      end do
      sum = sum*2.0d0

*     **** kx==0 plane, so single count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
            i=1
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + 1
            sum = sum + A(index)*B(index)
         end do
      end do

      
      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     ***** kx!=0 plane, so double count *****
      do index=1,nfft3d_map(nb)
            sum = sum + A(index)*B(index)
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do k=1,nz1(nb)
      do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(1,i,j,k,index,p)
         if (p.eq.taskid) then
         sum = sum - A(index)*B(index)
         end if
      end do
      end do

      end if
      

*     **** add up sums from other nodes ****
      if (np.gt.1) then
         call D3dB_SumAll(sum)
      end if

      sumall = sum
      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_tt_idot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_tt_idot(nb,A,B,sumall)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 B(*)
      real*8 sumall


#include "D3dB.fh"

      integer i,j,k,q,index,np,nxh,taskid,p
      real*8  sum


      call nwpw_timing_start(2)

      nxh=nx1(nb)/2
c      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ***** k!=0 plane, so double count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
         do i=2,(nxh+1)
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + i
            sum = sum + A(index)*B(index)
         end do
         end do
      end do
      sum = sum*2.0d0

*     **** kx==0 plane, so single count *****
      do q=1,nq(nb)
         do j=1,ny1(nb)
            i=1
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + 1
            sum = sum + A(index)*B(index)
         end do
      end do
      


      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     ***** kx!=0 plane, so double count *****
      do index=1,nfft3d_map(nb)
            sum = sum + A(index)*B(index)
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do k=1,nz1(nb)
      do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(1,i,j,k,index,p)
         if (p.eq.taskid) then
         sum = sum - A(index)*B(index)
         end if
      end do
      end do

      end if



*     **** !!!! do not add up sums from other nodes ****

      call nwpw_timing_end(2)

      sumall = sum
      return
      end






*     ***********************************
*     *					*
*     *	         D3dB_rr_dot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_dot(nb,A,B,sumall)     
      implicit none 
      integer nb
      real*8  A(*)
      real*8  B(*)
      real*8  sumall

#include "D3dB.fh"

      integer i,np
      real*8  sum

      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0
      do i=1,n2ft3d_map(nb)
         sum = sum + A(i)*B(i)
      end do

*     **** add up sums from other nodes ****
      if (np.gt.1) then
         call D3dB_SumAll(sum)
      end if

      sumall = sum
      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_rr_idot  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_idot(nb,A,B,sumall)     
      implicit none 
      integer nb
      real*8  A(*)
      real*8  B(*)
      real*8  sumall

#include "D3dB.fh"

      integer i,np
      real*8  sum


*     **** sum up dot product on this node ****
      sum = 0.0d0
      do i=1,n2ft3d_map(nb)
         sum = sum + A(i)*B(i)
      end do

*     **** add up sums from other nodes ****
*     if (np.gt.1) then
*        call D3dB_SumAll(sum)
*     end if

      sumall = sum
      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_cc_Mul  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_Mul(nb,A,B,C)     
      implicit none 
      integer    nb
      complex*16 A(*)
      complex*16 B(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
            C(i) = dconjg(A(i)) * B(i)
         end do

      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_lc_Mask  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_lc_Mask(nb,masker,A)
      implicit none 
      integer    nb
      logical    masker(*)
      complex*16 A(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         if (masker(i)) A(i) = dcmplx(0.0d0,0.0d0)
      end do
      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_lr_Mask  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_lr_Mask(nb,masker,A)
      implicit none 
      integer   nb
      logical   masker(*)
      real*8    A(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         if (masker(i)) A(i) = 0.0d0
      end do
      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_tc_Mul  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_tc_Mul(nb,A,B,C)     
      implicit none 
      integer    nb
      real*8     A(*)
      complex*16 B(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
            C(i) = A(i) * B(i)
      end do

      return
      end



*     ***********************************
*     *                                 *
*     *          D3dB_tc_Mul2           *
*     *                                 *
*     ***********************************

      subroutine D3dB_tc_Mul2(nb,A,B)
      implicit none
      integer    nb
      real*8     A(*)
      complex*16 B(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
            B(i) = B(i) * A(i)
      end do

      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_rr_Mul  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_Mul(nb,A,B,C)     
      implicit none 
      integer nb
      real*8 A(*)
      real*8 B(*)
      real*8 C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
            C(i) = A(i) * B(i)
         end do

      return
      end

*     ***********************************
*     *                                 *
*     *          D3dB_rr_Mul2           *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Mul2(nb,A,B)
      implicit none
      integer nb
      real*8 A(*)
      real*8 B(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         B(i) = B(i) * A(i)
      end do

      return
      end



*     ***********************************
*     *					*
*     *	         D3dB_cc_Sum  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_Sum(nb,A,B,C)     
      implicit none 
      integer    nb
      complex*16 A(*)
      complex*16 B(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = A(i) + B(i)
      end do

      return
      end


*     ***********************************
*     *                                 *
*     *          D3dB_cc_Sum2           *
*     *                                 *
*     ***********************************

      subroutine D3dB_cc_Sum2(nb,A,B)
      implicit none
      integer    nb
      complex*16 A(*)
      complex*16 B(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         B(i) = B(i) + A(i)
      end do

      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_rr_Sum  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_Sum(nb,A,B,C)     
      implicit none 
      integer nb
      real*8  A(*)
      real*8  B(*)
      real*8  C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = B(i)+A(i)
      end do

      return
      end


*     ***********************************
*     *                                 *
*     *          D3dB_rr_Sum2           *
*     *                                 *
*     ***********************************
      subroutine D3dB_rr_Sum2(nb,A,B)
      implicit none
      integer nb
      real*8  A(*)
      real*8  B(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         B(i) = B(i) + A(i)
      end do
      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_cc_Sub  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_Sub(nb,A,B,C)     
      implicit none 
      integer    nb
      complex*16 A(*)
      complex*16 B(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         C(i) = A(i) - B(i)
      end do

      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_rr_Sub   		*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_Sub(nb,A,B,C)     
      implicit none 
      integer nb
      real*8  A(*)
      real*8  B(*)
      real*8  C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = A(i) - B(i)
      end do

      return
      end



*     ***********************************
*     *                                 *
*     *          D3dB_rr_Sub2           *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Sub2(nb,A,B)
      implicit none
      integer nb
      real*8  A(*)
      real*8  B(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         B(i) = B(i) - A(i)
      end do

      return
      end




*     ***********************************
*     *					*
*     *	         D3dB_cc_zaxpy 	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_zaxpy(nb,alpha,A,B)     
      implicit none 
      integer    nb
      complex*16 alpha
      complex*16 A(*)
      complex*16 B(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         B(i) = B(i) + alpha*A(i)
      end do

      return
      end



*     ***********************************
*     *					*
*     *	         D3dB_cc_daxpy 	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_daxpy(nb,alpha,A,B)     
      implicit none 
      integer    nb
      real*8     alpha
      complex*16 A(*)
      complex*16 B(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
         B(i) = B(i) + alpha*A(i)
      end do

      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_rr_daxpy 	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_rr_daxpy(nb,alpha,A,B)     
      implicit none 
      integer nb
      real*8  alpha
      real*8  A(*)
      real*8  B(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         B(i) = B(i) + alpha* B(i)
      end do

      return
      end

*     ***********************************
*     *                                 *
*     *          D3dB_rr_Divide         *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Divide(nb,A,B,C)
      implicit none
      integer nb
      real*8 A(*)
      real*8 B(*)
      real*8 C(*)

#include "D3dB.fh"

      real*8 eta
      parameter (eta=1.0d-9)

      integer index

      !do q=1,nq(nb)
      !do j=1,ny1(nb)
      !do i=1,nx1(nb)
      do index = 1,n2ft3d_map(nb)
         !index = i + (j-1)*(nx1(nb)+2) + (q-1)*(nx1(nb)+2)*ny1(nb)
         if (dabs(B(index)) .le. eta) then
           C(index) = 0.0d0
         else
           C(index) = A(index) / B(index)
         end if
      end do
      !end do
      !end do
      !end do

      return
      end



*     ***********************************
*     *                                 *
*     *          D3dB_rr_Divide2         *
*     *                                 *
*     ***********************************

      subroutine D3dB_rr_Divide2(nb,A,B)
      implicit none
      integer nb
      real*8 A(*)
      real*8 B(*)

#include "D3dB.fh"

      real*8 eta
      parameter (eta=1.0d-9)

      integer index

      !do q=1,nq(nb)
      !do j=1,ny1(nb)
      !do i=1,nx1(nb)
      do index = 1,n2ft3d_map(nb)
         !index = i + (j-1)*(nx1(nb)+2) + (q-1)*(nx1(nb)+2)*ny1(nb)
         if (dabs(A(index)) .le. eta) then
           B(index) = 0.0d0
         else
           B(index) = B(index) / A(index)
         end if
      end do
      !end do
      !end do
      !end do

      return
      end



*     ***********************************
*     *                                 *
*     *          D3dB_r_ABS             *
*     *                                 *
*     ***********************************

      subroutine D3dB_r_ABS(nb,A,C)
      implicit none
      integer nb
      real*8 A(*)
      real*8 C(*)

#include "D3dB.fh"


      integer index

      !do q=1,nq(nb)
      !do j=1,ny1(nb)
      !do i=1,nx(nb)
      do index=1,n2ft3d_map(nb)
         !index = i + (j-1)*(nx(nb)+2) + (q-1)*(nx(nb)+2)*ny1(nb)
         C(index) = dabs(A(index))
      end do
      !end do
      !end do
      !end do

      return
      end

*     ***********************************
*     *                                 *
*     *          D3dB_r_ZeroNegative    *
*     *                                 *
*     ***********************************

      subroutine D3dB_r_ZeroNegative(nb,A)
      implicit none
      integer nb
      real*8 A(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         if (A(i).lt.0.0d0) A(i) = 0.0d0
      end do

      return
      end

*     ***********************************
*     *                                 *
*     *          D3dB_rr_Minus          *
*     *                                 *
*     ***********************************
      subroutine D3dB_rr_Minus(nb,A,B,C)
      implicit none
      integer nb
      real*8  A(*)
      real*8  B(*)
      real*8  C(*)

#include "D3dB.fh"

      integer i

      do i=1,n2ft3d_map(nb)
         C(i) = A(i) - B(i)
      end do

      return
      end




*     ***********************************
*     *					*
*     *          D3dB_r_Zero_Ends	*
*     *					*
*     ***********************************

      subroutine D3dB_r_Zero_Ends(nb,A)
      integer nb
      real*8 A(*) 

#include "D3dB.fh"

      integer j,k,q,index,taskid,p

      !**** slab mapping ****
      if (mapping.eq.1) then
      do q=1,nq(nb)
         do j=1,ny1(nb)
            index = (nx1(nb)+1) + (j-1)*(nx1(nb)+2) 
     >                         + (q-1)*(nx1(nb)+2)*(ny1(nb))
            A(index)   = 0.0d0
            A(index+1) = 0.0d0
         end do
      end do


      !**** hilbert mapping ****
      else
        call Parallel2d_taskid_i(taskid)
        do k=1,nz1(nb)
        do j=1,ny1(nb)

         call D3dB_ijktoindex2p(nb,(nx1(nb)+1),j,k,index,p)
         if (p.eq.taskid) A(index) = 0.0d0

         call D3dB_ijktoindex2p(nb,(nx1(nb)+2),j,k,index,p)
         if (p.eq.taskid) A(index) = 0.0d0

        end do
        end do
      end if

      return
      end


*     ***********************************
*     *					*
*     *	         D3dB_r_dsum  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_r_dsum(nb,A,sumall)     
      implicit none
      integer nb 
      real*8  A(*)
      real*8  sumall

#include "D3dB.fh"

      integer i,np
      real*8 sum

      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0
      do i=1,n2ft3d_map(nb)
         sum = sum + A(i)
      end do

*     **** add up sums from other nodes ****
      if (np.gt.1) then
        call D3dB_SumAll(sum)
      end if

      sumall = sum

      return
      end

*     ***********************************
*     *					*
*     *	         D3dB_t_dsum  	 	*	
*     *					*
*     ***********************************

      subroutine D3dB_t_dsum(nb,A,sumall)     
      implicit none 
      integer nb
      real*8  A(*)
      real*8  sumall

#include "D3dB.fh"

      integer i,j,k,q,np,nxh,index,taskid,p
      real*8 sum

      nxh = nx1(nb)/2
      call Parallel2d_np_i(np)

*     **** sum up dot product on this node ****
      sum = 0.0d0

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     ***** k!=0 plane so double count *****
      do q=1,nq(nb)
      do j=1,ny1(nb)
         do i=2,(nxh+1)
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + i
             sum = sum + A(index)
         end do
      end do
      end do
      sum = sum*2.0d0

*     ***** k==0 plane, so single count *****
      do q=1,nq(nb)
      do j=1,ny1(nb)
            index = (q-1)*(nxh+1)*ny1(nb) + (j-1)*(nxh+1) + 1
             sum = sum + A(index)
      end do
      end do


      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     ***** kx!=0 plane, so double count *****
      do index=1,nfft3d_map(nb)
            sum = sum + A(index)
      end do
      sum = sum*2.0d0

*     ***** kx==0 plane, so single count *****
      do k=1,nz1(nb)
      do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(nb,i,j,k,index,p)
         if (p.eq.taskid) then
         sum = sum - A(index)
         end if
      end do
      end do

      end if

*     **** add up sums from other nodes ****
      if (np.gt.1) then
        call D3dB_SumAll(sum)
      end if

      sumall = sum

      return
      end


*     ***********************************
*     *					*
*     *	     D3dB_cc_Vector_dot 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_Vector_dot(nb,nnfft3d,nn,ne,A,B,sumall)     
      implicit none 
      integer    nb
      integer    nnfft3d,nn,ne
      complex*16 A(*)
      complex*16 B(*)
      real*8     sumall(nn,nn)


#include "D3dB.fh"

      integer i,j,k,q,index,np,taskid,p
      integer index1,index2
      integer n,m,shift1,shift2
      real*8  sum


      call nwpw_timing_start(2)

      call Parallel2d_np_i(np)

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     **** sum up dot product on this node ****
      do n=1,ne
      do m=n,ne

        shift1 = (n-1)*nnfft3d
        shift2 = (m-1)*nnfft3d
        sum    = 0.0d0

*       ***** kx!=0 plane, so double count *****
        do q=1,nq(nb)
           do j=1,ny1(nb)
           do i=2,(nx1(nb)/2+1)
              index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >              + (j-1)*(nx1(nb)/2+1) + i
              index1 = index+shift1
              index2 = index+shift2
              sum = sum + dble(A(index1))  * dble(B(index2))
     >                  + dimag(A(index1)) * dimag(B(index2))
           end do
           end do
        end do
        sum = sum*2.0d0

*       ***** kx==0 plane, so single count *****
        do q=1,nq(nb)
           do j=1,ny1(nb)
              i=1
              index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >              + (j-1)*(nx1(nb)/2+1) + 1
              index1 = index+shift1
              index2 = index+shift2
              sum = sum + dble(A(index1))  * dble(B(index2))
     >                  + dimag(A(index1)) * dimag(B(index2))
           end do
        end do

        sumall(n,m) = sum
        sumall(m,n) = sum
      end do
      end do


      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     **** sum up dot product on this node ****
      do n=1,ne
      do m=n,ne

        shift1 = (n-1)*nnfft3d
        shift2 = (m-1)*nnfft3d
        sum    = 0.0d0

*       ***** kx!=0 plane, so double count *****
        do index=1,nfft3d_map(nb)
            index1 = index+shift1
            index2 = index+shift2
            sum = sum + dble(A(index1))  * dble(B(index2))
     >                + dimag(A(index1)) * dimag(B(index2))
        end do
        sum = sum*2.0d0

*       ***** kx==0 plane, so single count *****
        do k=1,nz1(nb)
        do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(nb,i,j,k,index,p)
         if (p.eq.taskid) then
         index1 = index+shift1
         index2 = index+shift2
         sum = sum - dble(A(index1))  * dble(B(index2))
     >             - dimag(A(index1)) * dimag(B(index2))
         end if
        end do
        end do


        sumall(n,m) = sum
        sumall(m,n) = sum
      end do
      end do

      end if


*     **** add up sums from other nodes ****
      if (np.gt.1) then
         call D3dB_Vector_SumAll(nn*ne,sumall)
      end if

      call nwpw_timing_end(2)
      return
      end



*     ***********************************
*     *					*
*     *	     D3dB_cc_Vector_ndot 	*	
*     *					*
*     ***********************************

      subroutine D3dB_cc_Vector_ndot(nb,nnfft3d,ne,A,B,sumall)     
      implicit none 
      integer    nb
      integer    nnfft3d,ne
      complex*16 A(*)
      complex*16 B(*)
      real*8     sumall(ne)


#include "D3dB.fh"

      integer i,j,k,q,index,np,taskid,p
      integer index1,index2
      integer n,shift1,shift2
      real*8  sum


      call nwpw_timing_start(2)

      call Parallel2d_np_i(np)

      !**********************
      !**** slab mapping ****
      !**********************
      if (mapping.eq.1) then
*     **** sum up dot product on this node ****
      do n=1,ne

        shift1 = (n-1)*nnfft3d
        shift2 = 0
        sum    = 0.0d0
*       ***** kx!=0 plane, so double count *****
        do q=1,nq(nb)
           do j=1,ny1(nb)
           do i=2,(nx1(nb)/2+1)
              index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >              + (j-1)*(nx1(nb)/2+1) + i
              index1 = index+shift1
              index2 = index+shift2
              sum = sum + dble(A(index1))  * dble(B(index2))
     >                  + dimag(A(index1)) * dimag(B(index2))
           end do
           end do
        end do
        sum = sum*2.0d0

*       ***** kx==0 plane, so single count *****
        do q=1,nq(nb)
           do j=1,ny1(nb)
              i=1
              index = (q-1)*(nx1(nb)/2+1)*ny1(nb) 
     >              + (j-1)*(nx1(nb)/2+1) + 1
              index1 = index+shift1
              index2 = index+shift2
              sum = sum + dble(A(index1))  * dble(B(index2))
     >                  + dimag(A(index1)) * dimag(B(index2))
           end do
        end do

         sumall(n) = sum
      end do


      !*************************
      !**** hilbert mapping ****
      !*************************
      else
      call Parallel2d_taskid_i(taskid)
*     **** sum up dot product on this node ****
      do n=1,ne

        shift1 = (n-1)*nnfft3d
        shift2 = 0
        sum    = 0.0d0

*       ***** kx!=0 plane, so double count *****
        do index=1,nfft3d_map(nb)
            index1 = index+shift1
            index2 = index+shift2
            sum = sum + dble(A(index1))  * dble(B(index2))
     >                + dimag(A(index1)) * dimag(B(index2))
        end do
        sum = sum*2.0d0

*       ***** kx==0 plane, so single count *****
        do k=1,nz1(nb)
        do j=1,ny1(nb)
         i=1
         call D3dB_ijktoindexp(nb,i,j,k,index,p)
         if (p.eq.taskid) then
         index1 = index+shift1
         index2 = index+shift2
         sum = sum - dble(A(index1))  * dble(B(index2))
     >             - dimag(A(index1)) * dimag(B(index2))
         end if
        end do
        end do

        sumall(n) = sum
      end do

      end if


*     **** add up sums from other nodes ****
      if (np.gt.1) then
         call D3dB_Vector_SumAll(ne,sumall)
      end if

      call nwpw_timing_end(2)

      return
      end


*     ***********************************
*     *                                 *
*     *          D3dB_ic_Mul            *
*     *                                 *
*     ***********************************
cpgi$r opt=1
      subroutine D3dB_ic_Mul(nb,A,B,C)
      implicit none
      integer    nb
      real*8     A(*)
      complex*16 B(*)
      complex*16 C(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
            C(i) = dcmplx(0.0d0,A(i)) * B(i)
      end do

      return
      end

*     ***********************************
*     *                                 *
*     *          D3dB_ic_Mul2           *
*     *                                 *
*     ***********************************
cpgi$r opt=1
      subroutine D3dB_ic_Mul2(nb,A,B)
      implicit none
      integer    nb
      real*8     A(*)
      complex*16 B(*)

#include "D3dB.fh"

      integer i

      do i=1,nfft3d_map(nb)
            B(i) = dcmplx(0.0d0,A(i)) * B(i)
      end do

      return
      end



      subroutine D3dB_pfft_index1_copy(n,index,a,b)
      implicit none
      integer n
      integer index(*)
      complex*16  a(*),b(*)
      integer i
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,n
        b(i) = a(index(i))
      end do
      return
      end

      subroutine D3dB_pfft_index2_copy(n,index,a,b)
      implicit none
      integer n
      integer index(*)
      complex*16  a(*),b(*)
      integer i
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,n
        b(index(i)) = a(i)
      end do
      return
      end

      subroutine D3dB_pfft_index2_zero(n,index,a)
      implicit none
      integer n
      integer index(*)
      complex*16  a(*)
      integer i
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,n
        a(index(i)) = 0.0d0
      end do
      return
      end


*     ***********************************
*     *					*
*     *	   D3dB_c_transpose_jk		*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_jk(nb,A,tmp1,tmp2)

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
      integer nb
      complex*16  A(*)
      complex*16  tmp1(*),tmp2(*)

#include "mpif.h"
#include "D3dB.fh"

*     **** indexing variables ****
      integer iq_to_i1(nfft3d_max,NBLOCKS)
      integer iq_to_i2(nfft3d_max,NBLOCKS)
      integer i1_start(nz+1,NBLOCKS)
      integer i2_start(nz+1,NBLOCKS)
      common / trans_blk / iq_to_i1,iq_to_i2,i1_start,i2_start



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
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,nfft3d(nb) 
         tmp1(iq_to_i1(i,nb)) = A(i)
      end do

*     **** it = 0, transpose data on same thread ****  
      msglen = i2_start(2,nb) - i2_start(1,nb)
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,msglen
         tmp2(i2_start(1,nb)+i-1) = tmp1(i1_start(1,nb)+i-1)
      end do
         

*     **** receive packed array data ****
      reqcnt = 0
      do it = 1,np-1

*        **** synchronous receive of tmp ****
         proc_from = mod((taskid)-it+np, np) 

         msgtype = 1
         source=proc_from
         type   = 1
         msglen = i2_start(it+2,nb) - i2_start(it+1,nb)
         
         if (msglen.gt.0) then
            reqcnt = reqcnt + 1
            call MPI_IRECV(tmp2(i2_start(it+1,nb)),
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
         msglen    = i1_start(it+2,nb) - i1_start(it+1,nb)
         msgtype   = 1
      
         if (msglen.gt.0) then
            call MPI_SEND(tmp1(i1_start(it+1,nb)),
     >                     msglen,MPI_DOUBLE_COMPLEX,
     >                     Parallel2d_convert_taskid_i(proc_to),
     >                     msgtype,MPI_COMM_WORLD,mpierr)
         end if

      end do

   
*     **** wait for completion of mp_send, also do a sync ****
      if (np.gt.1) call Parallel_mpiWaitAll(reqcnt,request)


*     **** unpack A(i) array ****
#ifndef CRAY
!DIR$ ivdep
#endif
      do i=1,nfft3d(nb) !***(nx(nb)/2+1)*ny1(nb)*nq(nb)
         A(i) = tmp2(iq_to_i2(i,nb))
      end do

      return
      end




*     ***********************************
*     *					*
*     *	   D3dB_c_transpose_ijk		*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_ijk(nb,op,A,tmp1,tmp2,nfft3d0)

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
      integer nb,op,nfft3d0
      complex*16  A(nfft3d0)
      complex*16  tmp1(nfft3d0),tmp2(nfft3d0)

#include "D3dB.fh"
#include "mpif.h"


*     **** indexing variables ****
      integer h_iq_to_i1(nfft3d_max,6,NBLOCKS)
      integer h_iq_to_i2(nfft3d_max,6,NBLOCKS)
      integer h_i1_start(np_max+1,6,NBLOCKS)
      integer h_i2_start(np_max+1,6,NBLOCKS)
      common / trans_blk_ijk / h_iq_to_i1,
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
      if ((op.eq.1).or.(op.eq.5)) nnfft3d = (nx1(nb)/2+1)*nq1(nb)
      if ((op.eq.2).or.(op.eq.4)) nnfft3d = (ny1(nb))    *nq2(nb)
      if ((op.eq.3).or.(op.eq.6)) nnfft3d = (nz1(nb))    *nq3(nb)

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,nnfft3d
         tmp1(h_iq_to_i1(i,op,nb)) = A(i)
      end do
!$OMP END PARALLEL DO


*     **** it = 0, transpose data on same thread ****  
      msglen = h_i2_start(2,op,nb)
     >       - h_i2_start(1,op,nb)
      call dcopy(2*msglen,tmp1(h_i1_start(1,op,nb)),1,
     >                    tmp2(h_i2_start(1,op,nb)),1)

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
         msglen = h_i2_start(it+2,op,nb) - h_i2_start(it+1,op,nb)
         

         if (msglen.gt.0) then
            reqcnt = reqcnt + 1
            call MPI_IRECV(tmp2(h_i2_start(it+1,op,nb)),
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
         msglen    = h_i1_start(it+2,op,nb)- h_i1_start(it+1,op,nb)
         msgtype   = 1
      
         if (msglen.gt.0) then
            call MPI_SEND(tmp1(h_i1_start(it+1,op,nb)),
     >                     msglen,MPI_DOUBLE_COMPLEX,
     >                     Parallel2d_convert_taskid_i(proc_to),
     >                     msgtype,MPI_COMM_WORLD,mpierr)
         end if

      end do

   
*     **** wait for completion of mp_send, also do a sync ****
      if (np.gt.1) call Parallel_mpiWaitAll(reqcnt,request)


*     **** unpack A(i) array ****
      if ((op.eq.4).or.(op.eq.6)) nnfft3d = (nx1(nb)/2+1)*nq1(nb)
      if ((op.eq.1).or.(op.eq.3)) nnfft3d = (ny1(nb))    *nq2(nb)
      if ((op.eq.2).or.(op.eq.5)) nnfft3d = (nz1(nb))    *nq3(nb)

!$OMP PARALLEL DO private(i) schedule(dynamic,100)
      do i=1,nnfft3d
         A(i) = tmp2(h_iq_to_i2(i,op,nb))
      end do 
!$OMP END PARALLEL DO

      return
      end




*     ***********************************
*     *                                 *
*     *         D3dB_SumAll             *
*     *                                 *
*     ***********************************

      subroutine D3dB_SumAll(sum)
c     implicit none
      real*8  sum

#include "mpif.h"

      integer msglen,mpierr,np
      real*8 sumall

*     **** external functions ****
      integer  Parallel2d_comm_i
      external Parallel2d_comm_i

      call Parallel2d_np_i(np)
      if (np.gt.1) then
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_SUM,Parallel2d_comm_i(),mpierr)
         sum = sumall
      end if

      return
      end



*     ***********************************
*     *                                 *
*     *         D3dB_ISumAll            *
*     *                                 *
*     ***********************************

      subroutine D3dB_ISumAll(sum)
c     implicit none
      integer sum

#include "mpif.h"

      integer msglen,mpierr,np
      integer sumall

*     **** external functions ****
      integer  Parallel2d_comm_i
      external Parallel2d_comm_i

      call Parallel2d_np_i(np)
      if (np.gt.1) then

         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_INTEGER,
     >                       MPI_SUM,Parallel2d_comm_i(),mpierr)
         sum = sumall
      end if

      return
      end





*     ***********************************
*     *                                 *
*     *         D3dB_Vector_SumAll      *
*     *                                 *
*     ***********************************

      subroutine D3dB_Vector_SumAll(n,sum)
c     implicit none
      integer n
      real*8  sum(*)

#include "mpif.h"

      logical value
      integer msglen,mpierr

*     **** temporary workspace ****
      integer np,i
      real*8 sumall(n)

*     **** external functions ****
      integer  Parallel2d_comm_i
      external Parallel2d_comm_i

      call Parallel2d_np_i(np)
      call nwpw_timing_start(2)
      if (np.gt.1) then

*     ***** allocate temporary space ****
      call MPI_Allreduce(sum,sumall,n,
     >                MPI_DOUBLE_PRECISION,
     >                MPI_SUM,Parallel2d_comm_i(),mpierr)

      do i=1,n
         sum(i) = sumall(i)
      end do

      end if
      call nwpw_timing_end(2)
      return
      end





*     ***********************************
*     *					*
*     *	   D3dB_c_transpose_ijk_init	*
*     *					*
*     ***********************************

      subroutine D3dB_c_transpose_ijk_init(nb)
      implicit none
      integer nb

#include "D3dB.fh"


*     *** hilbert tranpose data structure ****
      integer h_iq_to_i1(nfft3d_max,6,NBLOCKS)
      integer h_iq_to_i2(nfft3d_max,6,NBLOCKS)
      integer h_i1_start(np_max+1,6,NBLOCKS)
      integer h_i2_start(np_max+1,6,NBLOCKS)
      common / trans_blk_ijk / h_iq_to_i1,
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
         h_i1_start(it+1,1,nb) = index1
         h_i2_start(it+1,1,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme **** 
            phere = p_map1((j-1)+(k-1)*ny1(nb)+1,nb)
            qhere = q_map1((j-1)+(k-1)*ny1(nb)+1,nb)

            pto   = p_map2((k-1)+(i-1)*nz1(nb)+1,nb)
            qto   = q_map2((k-1)+(i-1)*nz1(nb)+1,nb)
            
           
            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = i + (qhere-1)*(nx1(nb)/2+1)
               h_iq_to_i1(itmp,1,nb) = index1
               index1 = index1 + 1
            end if
             
*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = j + (qto-1)*ny1(nb)
               h_iq_to_i2(itmp,1,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,1,nb) = index1
      h_i2_start(np+1,1,nb) = index2





*     *********************************************************
*     **** map2to3 mapping - done - transpose operation #2 ****
*     *********************************************************

*     **** allocate trans_blk_ijk common block ****
      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,2,nb) = index1
         h_i2_start(it+1,2,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme ****
            phere = p_map2((k-1)+(i-1)*nz1(nb)+1,nb)
            qhere = q_map2((k-1)+(i-1)*nz1(nb)+1,nb)

            pto   = p_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)
            qto   = q_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)

            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = j + (qhere-1)*ny1(nb)
               h_iq_to_i1(itmp,2,nb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = k + (qto-1)*nz1(nb)
               h_iq_to_i2(itmp,2,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,2,nb) = index1
      h_i2_start(np+1,2,nb) = index2






*     ********************************************************
*     **** map3to2 mapping - done - tranpose operation #3 ****
*     ********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,3,nb) = index1
         h_i2_start(it+1,3,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme ****
            phere = p_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)
            qhere = q_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)

            pto   = p_map2((k-1)+(i-1)*nz1(nb)+1,nb)
            qto   = q_map2((k-1)+(i-1)*nz1(nb)+1,nb)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = k + (qhere-1)*nz1(nb)
               h_iq_to_i1(itmp,3,nb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = j + (qto-1)*ny1(nb)
               h_iq_to_i2(itmp,3,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,3,nb) = index1
      h_i2_start(np+1,3,nb) = index2




*     ********************************************************
*     **** map2to1 mapping - done - tranpose operation #4 ****
*     ********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,4,nb) = index1
         h_i2_start(it+1,4,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme ****
            phere = p_map2((k-1)+(i-1)*nz1(nb)+1,nb)
            qhere = q_map2((k-1)+(i-1)*nz1(nb)+1,nb)

            pto   = p_map1((j-1)+(k-1)*ny1(nb)+1,nb)
            qto   = q_map1((j-1)+(k-1)*ny1(nb)+1,nb)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = j + (qhere-1)*ny1(nb)
               h_iq_to_i1(itmp,4,nb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = i + (qto-1)*(nx1(nb)/2+1)
               h_iq_to_i2(itmp,4,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,4,nb) = index1
      h_i2_start(np+1,4,nb) = index2





*     **********************************************************
*     **** map1to3 mapping  - done - tranpose operation # 5 ****
*     **********************************************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,5,nb) = index1
         h_i2_start(it+1,5,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme ****
            phere = p_map1((j-1)+(k-1)*ny1(nb)+1,nb)
            qhere = q_map1((j-1)+(k-1)*ny1(nb)+1,nb)

            pto   = p_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)
            qto   = q_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = i + (qhere-1)*(nx1(nb)/2+1)
               h_iq_to_i1(itmp,5,nb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = k + (qto-1)*nz1(nb)
               h_iq_to_i2(itmp,5,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,5,nb) = index1
      h_i2_start(np+1,5,nb) = index2






*     *************************
*     **** map3to1 mapping ****
*     *************************

      index1 = 1
      index2 = 1
      do it=0,np-1
         proc_to   = mod(taskid+it,np)
         proc_from = mod(taskid-it+np,np)
         h_i1_start(it+1,6,nb) = index1
         h_i2_start(it+1,6,nb) = index2

         do k=1,nz1(nb)
         do j=1,ny1(nb)
         do i=1,(nx1(nb)/2+1)

*           **** packing scheme ****
            phere = p_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)
            qhere = q_map3((i-1)+(j-1)*(nx1(nb)/2+1)+1,nb)

            pto   = p_map1((j-1)+(k-1)*ny1(nb)+1,nb)
            qto   = q_map1((j-1)+(k-1)*ny1(nb)+1,nb)


            if ((phere.eq.taskid).and.(pto.eq.proc_to)) then
               itmp = k + (qhere-1)*nz1(nb)
               h_iq_to_i1(itmp,6,nb) = index1
               index1 = index1 + 1
            end if

*           **** unpacking scheme ****
            if ((pto.eq.taskid).and.(phere.eq.proc_from)) then
               itmp = i + (qto-1)*(nx1(nb)/2+1)
               h_iq_to_i2(itmp,6,nb) = index2
               index2 = index2 + 1
            end if

         end do
         end do
         end do

      end do
      h_i1_start(np+1,6,nb) = index1
      h_i2_start(np+1,6,nb) = index2

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


