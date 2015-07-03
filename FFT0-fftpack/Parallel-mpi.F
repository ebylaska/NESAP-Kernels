* Parallel.f
* Author - Eric Bylaska
*
*   These routines are to be used to keep track of the parallel message
* passing variables, as well as iniitialize and deinitialize the
* message passing routines.
*

*     *************************************
*     *                                   *
*     *        Parallel_Init              *
*     *                                   *
*     *************************************

      subroutine Parallel_Init()
      implicit none

#include "Parallel.fh"
#include "mpif.h"

*     **** local variables ****
      integer i,mpierr

*     **** MPI initiializer *****
      call MPI_INIT(mpierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,taskid,mpierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,np,mpierr)

*     **** set up 3d processor grid = np x 1 x 1****
      np_i = np
      np_j = 1
      np_k = 1
      do i=0,np-1
        procNd(i+1) = i
      end do
      taskid_i = taskid
      taskid_j = 0
      taskid_k = 0
      comm_i   = MPI_COMM_WORLD
      comm_j   = -99 
      comm_k   = -99 

      return 
      end



*     *************************************
*     *                                   *
*     *        Parallel2d_Init            *
*     *                                   *
*     *************************************

*     Sset up the 2d processor grid = np_i x np_j, 
*     where np_i = nrows, and np_j = np/np_i
*
      subroutine Parallel2d_Init(ncolumns)
      implicit none
      integer ncolumns

#include "Parallel.fh"
#include "mpif.h"

*     *** local variables ***
      integer i,j,icount,ierr
      integer tmp(np_max),tmp2(np_max),mpi_group

      if (ncolumns.gt.1) then

      np_i = np/ncolumns
      np_j = ncolumns


      icount = 0
      do j=0,np_j-1
      do i=0,np_i-1
        if (icount.eq.taskid) then
           taskid_i = i
           taskid_j = j
        end if
        procNd(1+i+j*np_i) = icount
        icount = mod((icount+1),np)
      end do
      end do

*     **** set global processor group ****
      call MPI_COMM_group(MPI_COMM_WORLD,mpi_group,ierr)

      do i=0,np_i-1
        tmp(1+i) = procNd(1+i+taskid_j*np_i) 
      end do
      call MPI_Group_incl(mpi_group,np_i,tmp,group_i,ierr)
      call MPI_Comm_create(MPI_COMM_WORLD,group_i,comm_i,ierr)

      do j=0,np_j-1
        tmp(1+j) = procNd(1+taskid_i+j*np_i) 
      end do
      call MPI_Group_incl(mpi_group,np_j,tmp,group_j,ierr)
      call MPI_Comm_create(MPI_COMM_WORLD,group_j,comm_j,  ierr)

      end if
      return
      end


*     *************************************
*     *                                   *
*     *        Parallel2d_Finalize        *
*     *                                   *
*     *************************************

      subroutine Parallel2d_Finalize()
      implicit none

#include "Parallel.fh"
#include "mpif.h"

*     *** local variable ***
      integer mpierr

      if (np_j.gt.1) then
*      **** free comm_i and comm_j communicators ****
      call MPI_Comm_free(comm_i,  mpierr)
      call MPI_Group_free(group_i,mpierr)
      call MPI_Comm_free(comm_j,  mpierr)
      call MPI_Group_free(group_j,mpierr)
      end if
      return
      end



*     *************************************
*     *                                   *
*     *        Parallel3d_Init            *
*     *                                   *
*     *************************************

*     Sset up the 3d processor grid = np_i x np_j x np_k, 
*     where np_i = nrows=np/(np_j*np_k), 
*     np_j = ncolumns, and np_k = nzones
*
      subroutine Parallel3d_Init(ncolumns,nzones)
      implicit none
      integer ncolumns,nzones

#include "Parallel.fh"
#include "mpif.h"

*     *** local variables ***
      integer i,j,k,icount,ierr
      integer tmp(np_max),tmp2(np_max),mpi_group

      np_i = np/(ncolumns*nzones)
      np_j = ncolumns
      np_k = nzones

      icount = 0
      do k=0,np_k-1
      do j=0,np_j-1
      do i=0,np_i-1
        if (icount.eq.taskid) then
           taskid_i = i
           taskid_j = j
           taskid_k = k
        end if
        procNd(1 + i + j*np_i + k*np_i*np_j) = icount
        icount = mod((icount+1),np)
      end do
      end do
      end do


*     **** set global processor group ****
      call MPI_COMM_group(MPI_COMM_WORLD,mpi_group,ierr)

      do i=0,np_i-1
        tmp(1+i) = procNd(1
     >                   + i 
     >                   + taskid_j*np_i 
     >                   + taskid_k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_i,tmp,group_i,ierr)
      call MPI_Comm_create(MPI_COMM_WORLD,group_i,comm_i,ierr)

      do j=0,np_j-1
        tmp(1+j) = procNd(1 
     >                   + taskid_i 
     >                   + j*np_i
     >                   + taskid_k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_j,tmp,group_j,ierr)
      call MPI_Comm_create(MPI_COMM_WORLD,group_j,comm_j,ierr)

      do k=0,np_k-1
        tmp(1+k) = procNd(1 
     >                   + taskid_i 
     >                   + taskid_j*np_i
     >                   + k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_k,tmp,group_k,ierr)
      call MPI_Comm_create(MPI_COMM_WORLD,group_k,comm_k,ierr)

      return
      end


*     *************************************
*     *                                   *
*     *        Parallel3d_Finalize        *
*     *                                   *
*     *************************************

      subroutine Parallel3d_Finalize()
      implicit none

#include "Parallel.fh"
#include "mpif.h"

*     *** local variable ***
      integer mpierr

*      **** free comm_i and comm_j communicators ****
      call MPI_Comm_free(comm_i,  mpierr)
      call MPI_Group_free(group_i,mpierr)
      call MPI_Comm_free(comm_j,  mpierr)
      call MPI_Group_free(group_j,mpierr)
      call MPI_Comm_free(comm_k,  mpierr)
      call MPI_Group_free(group_k,mpierr)
      return
      end


*     ***********************************
*     *                                 *
*     *         Parallel_MaxAll         *
*     *                                 *
*     ***********************************

      subroutine Parallel_MaxAll(sum)
c     implicit none
      real*8  sum

#include "Parallel.fh"
#include "mpif.h"

      real*8 sumall
      integer msglen,mpierr

      if (np.gt.1) then
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_MAX,MPI_COMM_WORLD,mpierr)
         sum = sumall
      end if
      return
      end





*     ***********************************
*     *                                 *
*     *         Parallel_SumAll         *
*     *                                 *
*     ***********************************

      subroutine Parallel_SumAll(sum)
c     implicit none
      real*8  sum


#include "Parallel.fh"
#include "mpif.h"

      real*8 sumall
      integer msglen,mpierr

      if (np.gt.1) then
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_SUM,MPI_COMM_WORLD,mpierr)
         sum = sumall
      end if

      return
      end


*     ***********************************
*     *                                 *
*     *         Parallel_ISumAll        *
*     *                                 *
*     ***********************************

      subroutine Parallel_ISumAll(sum)
c     implicit none
      integer sum


#include "Parallel.fh"
#include "mpif.h"

      integer msglen,mpierr
      integer sumall

      if (np.gt.1) then
         msglen = 1

         call MPI_Allreduce(sum,sumall,msglen,MPI_INTEGER,
     >                       MPI_SUM,MPI_COMM_WORLD,mpierr)
         sum = sumall
      end if

      return
      end

*     ***********************************
*     *                                 *
*     *      Parallel_Vector_SumAll     *
*     *                                 *
*     ***********************************

      subroutine Parallel_Vector_SumAll(n,sum)
c     implicit none
      integer n
      real*8  sum(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variable ****
      logical value
      integer msglen,mpierr,i

*     **** temporary workspace ****
      real*8 sumall(np_max)


      call nwpw_timing_start(2)
      if (np.gt.1) then


      msglen = n
      call MPI_Allreduce(sum,sumall,msglen,
     >                MPI_DOUBLE_PRECISION,
     >                MPI_SUM,MPI_COMM_WORLD,mpierr)
c      call dcopy(n,sumall,1,sum,1)
      do i=1,n
         sum(i) = sumall(i)
      end do

      end if
      call nwpw_timing_end(2)
      return
      end


*     ***********************************
*     *                                 *
*     *      Parallel_Vector_ISumAll    *
*     *                                 *
*     ***********************************

      subroutine Parallel_Vector_ISumAll(n,sum)
c     implicit none
      integer n
      integer  sum(*)

#include "Parallel.fh"
#include "mpif.h"

      logical value
      integer msglen,mpierr

*     **** temporary workspace ****
      integer sumall(np_max),sumall1(np_max)


      call nwpw_timing_start(2)

      if (np.gt.1) then

*     ***** allocate temporary space ****

      msglen = n
      call MPI_Allreduce(sum,sumall,msglen,
     >                MPI_INTEGER,
     >                MPI_SUM,MPI_COMM_WORLD,mpierr)
      call icopy(n,sumall,1,sum,1)
      end if

      call nwpw_timing_end(2)
      return
      end





*     ***********************************
*     *                                 *
*     *      Parallel_Brdcst_value      *
*     *                                 *
*     ***********************************

      subroutine Parallel_Brdcst_value(psend,sum)
      implicit none
      integer psend
      real*8  sum

#include "Parallel.fh"
#include "mpif.h"

      integer ierr

      if (np.gt.1) then
         call MPI_Bcast(sum,1,MPI_DOUBLE_PRECISION,
     >                  psend,MPI_COMM_WORLD,ierr)
      end if
      return
      end


*     ***********************************
*     *                                 *
*     *      Parallel_Brdcst_values     *
*     *                                 *
*     ***********************************

      subroutine Parallel_Brdcst_values(psend,nsize,sum)
      implicit none
      integer psend,nsize
      real*8  sum(*)

#include "Parallel.fh"
#include "mpif.h"

      integer ierr

      if (np.gt.1) then
         call MPI_Bcast(sum,nsize,MPI_DOUBLE_PRECISION,
     >                  psend,MPI_COMM_WORLD,ierr)
      end if
      return
      end




*     ***********************************
*     *                                 *
*     *      Parallel_Brdcst_ivalues    *
*     *                                 *
*     ***********************************

      subroutine Parallel_Brdcst_ivalues(psend,nsize,isum)
      implicit none
      integer psend,nsize
      integer isum(*)

#include "Parallel.fh"
#include "mpif.h"

      integer ierr

      if (np.gt.1) then
         call MPI_Bcast(isum,nsize,MPI_INTEGER,
     >                  psend,MPI_COMM_WORLD,ierr)
      end if
      return
      end


      subroutine icopy(n,array1,istr1,array2,istr2)
      implicit none
      integer n,istr1,istr2
      integer array1(*),array2(*)
      integer i,i1,i2
      i1 = 1
      i2 = 1
      do i=1,n
         array2(i2) = array1(i1)
         i1 = i1 + istr1 
         i2 = i2 + istr2 
      end do
      return
      end



*     ***********************************
*     *                                 *
*     *      Parallel_mpiWaitAll        *
*     *                                 *
*     ***********************************

      subroutine Parallel_mpiWaitAll(nreq,req)
      implicit none
      integer nreq,req(*)

#include "mpif.h"
#include "Parallel.fh"

*     *** local variables ***
      integer status(MPI_STATUS_SIZE*4*np_max),mpierr
     
      if (nreq.gt.0) then

*     **** wait for completion of mp_send, also do a sync ****
      call MPI_WAITALL(nreq,req,status,mpierr)
c      call MPI_WAITALL(nreq,req,MPI_STATUSES_IGNORE,mpierr)

*     *** may need to check status here??? ***


      end if
      return
      end



*     *************************************
*     *                                   *
*     *        Parallel_Finalize          *
*     *                                   *
*     *************************************

      subroutine Parallel_Finalize()
      implicit none

#include "Parallel.fh"
#include "mpif.h"

      integer mpierr
      call MPI_FINALIZE(mpierr)
      return
      end


*     *************************************
*     *                                   *
*     *        Parallel_np                *
*     *                                   *
*     *************************************

      subroutine Parallel_np(np_out)
      implicit none
      integer np_out

#include "Parallel.fh"

      np_out = np
      return
      end


*     *************************************
*     *                                   *
*     *        Parallel_taskid            *
*     *                                   *
*     *************************************

      subroutine Parallel_taskid(task_out)
      implicit none
      integer task_out
      
#include "Parallel.fh"

      task_out = taskid
      return 
      end




*     *************************************
*     *                                   *
*     *        Parallel2d_np_i            *
*     *                                   *
*     *************************************
      subroutine Parallel2d_np_i(np_out)
      implicit none
      integer np_out

#include  "Parallel.fh"

      np_out = np_i
      return
      end

*     *************************************
*     *                                   *
*     *        Parallel2d_np_j            *
*     *                                   *
*     *************************************
      subroutine Parallel2d_np_j(np_out)
      implicit none
      integer np_out

#include  "Parallel.fh"

      np_out = np_j
      return
      end


*     *************************************
*     *                                   *
*     *        Parallel2d_taskid_i        *
*     *                                   *
*     *************************************
      subroutine Parallel2d_taskid_i(taskid_out)
      implicit none
      integer taskid_out

#include "Parallel.fh"

      taskid_out = taskid_i
      return
      end 

*     *************************************
*     *                                   *
*     *        Parallel2d_taskid_j        *
*     *                                   *
*     *************************************
      subroutine Parallel2d_taskid_j(taskid_out)
      implicit none
      integer taskid_out

#include "Parallel.fh"

      taskid_out = taskid_j
      return
      end


*     *************************************
*     *                                   *
*     *     Parallel2d_convert_taskid_i   *
*     *                                   *
*     *************************************
      integer function Parallel2d_convert_taskid_i(i)
      implicit none
      integer i

#include "Parallel.fh"

      Parallel2d_convert_taskid_i = procNd(1
     >                                     + i
     >                                     + taskid_j*np_i 
     >                                     + taskid_k*np_i*np_j)
      return
      end

*     *************************************
*     *                                   *
*     *     Parallel2d_convert_taskid_j   *
*     *                                   *
*     *************************************
      integer function Parallel2d_convert_taskid_j(j)
      implicit none
      integer j

#include "Parallel.fh"

      Parallel2d_convert_taskid_j = procNd(1
     >                                    + taskid_i
     >                                    + j*np_i 
     >                                    + taskid_k*np_i*np_j)
      return
      end




*     *************************************
*     *                                   *
*     *   Parallel2d_convert_taskid_ij    *
*     *                                   *
*     *************************************
      integer function Parallel2d_convert_taskid_ij(i,j)
      implicit none
      integer i,j

#include "Parallel.fh"

      Parallel2d_convert_taskid_ij = procNd(1+i+j*np_i 
     >                                    + taskid_k*np_i*np_j)
      return
      end



*     *************************************
*     *                                   *
*     *         Parallel2d_comm_i         *
*     *                                   *
*     *************************************
      integer function Parallel2d_comm_i()
      implicit none

#include "Parallel.fh"

      Parallel2d_comm_i = comm_i
      return
      end

*     *************************************
*     *                                   *
*     *         Parallel2d_comm_j         *
*     *                                   *
*     *************************************
      integer function Parallel2d_comm_j()
      implicit none

#include "Parallel.fh"

      Parallel2d_comm_j = comm_j
      return
      end




*     *************************************
*     *                                   *
*     *        Parallel3d_np_i            *
*     *                                   *
*     *************************************
      subroutine Parallel3d_np_i(np_out)
      implicit none
      integer np_out

#include  "Parallel.fh"

      np_out = np_i
      return
      end

*     *************************************
*     *                                   *
*     *        Parallel3d_np_j            *
*     *                                   *
*     *************************************
      subroutine Parallel3d_np_j(np_out)
      implicit none
      integer np_out

#include  "Parallel.fh"

      np_out = np_j
      return
      end

*     *************************************
*     *                                   *
*     *        Parallel3d_np_k            *
*     *                                   *
*     *************************************
      subroutine Parallel3d_np_k(np_out)
      implicit none
      integer np_out

#include  "Parallel.fh"

      np_out = np_k
      return
      end


*     *************************************
*     *                                   *
*     *        Parallel3d_taskid_i        *
*     *                                   *
*     *************************************
      subroutine Parallel3d_taskid_i(taskid_out)
      implicit none
      integer taskid_out

#include "Parallel.fh"

      taskid_out = taskid_i
      return
      end 

*     *************************************
*     *                                   *
*     *        Parallel3d_taskid_j        *
*     *                                   *
*     *************************************
      subroutine Parallel3d_taskid_j(taskid_out)
      implicit none
      integer taskid_out

#include "Parallel.fh"

      taskid_out = taskid_j
      return
      end

*     *************************************
*     *                                   *
*     *        Parallel3d_taskid_k        *
*     *                                   *
*     *************************************
      subroutine Parallel3d_taskid_k(taskid_out)
      implicit none
      integer taskid_out

#include "Parallel.fh"

      taskid_out = taskid_k
      return
      end



*     *************************************
*     *                                   *
*     *     Parallel3d_convert_taskid_i   *
*     *                                   *
*     *************************************
      integer function Parallel3d_convert_taskid_i(i)
      implicit none
      integer i

#include "Parallel.fh"

      Parallel3d_convert_taskid_i = procNd(1
     >                                    + i
     >                                    + taskid_j*np_i 
     >                                    + taskid_k*np_i*np_j)
      return
      end

*     *************************************
*     *                                   *
*     *     Parallel3d_convert_taskid_j   *
*     *                                   *
*     *************************************
      integer function Parallel3d_convert_taskid_j(j)
      implicit none
      integer j

#include "Parallel.fh"

      Parallel3d_convert_taskid_j = procNd(1
     >                                    + taskid_i
     >                                    + j*np_i
     >                                    + taskid_k*np_i*np_j)
      return
      end

*     *************************************
*     *                                   *
*     *     Parallel3d_convert_taskid_k   *
*     *                                   *
*     *************************************
      integer function Parallel3d_convert_taskid_k(k)
      implicit none
      integer k

#include "Parallel.fh"

      Parallel3d_convert_taskid_k = procNd(1
     >                                    + taskid_i
     >                                    + taskid_j*np_i
     >                                    + k*np_i*np_j)
      return
      end



*     *************************************
*     *                                   *
*     *   Parallel3d_convert_taskid_ijk   *
*     *                                   *
*     *************************************
      integer function Parallel3d_convert_taskid_ijk(i,j,k)
      implicit none
      integer i,j,k

#include "Parallel.fh"

      Parallel3d_convert_taskid_ijk = procNd(1
     >                                      + i
     >                                      + j*np_i
     >                                      + k*np_i*np_j)

      return
      end



*     *************************************
*     *                                   *
*     *         Parallel3d_comm_i         *
*     *                                   *
*     *************************************
      integer function Parallel3d_comm_i()
      implicit none

#include "Parallel.fh"

      Parallel3d_comm_i = comm_i
      return
      end

*     *************************************
*     *                                   *
*     *         Parallel3d_comm_j         *
*     *                                   *
*     *************************************
      integer function Parallel3d_comm_j()
      implicit none

#include "Parallel.fh"

      Parallel3d_comm_j = comm_j
      return
      end

*     *************************************
*     *                                   *
*     *         Parallel3d_comm_k         *
*     *                                   *
*     *************************************
      integer function Parallel3d_comm_k()
      implicit none

#include "Parallel.fh"

      Parallel3d_comm_k = comm_k
      return
      end


