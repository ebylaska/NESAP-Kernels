*
* $Id: Parallel-mpi.F 26777 2015-01-30 18:59:15Z mjacquelin $
*

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
      integer MASTER
      parameter (MASTER=0)
      integer i,mpierr,mygroup,myio,mydepth,provided


c*    **** MPI initializer - for openmp MPI_THREAD_SERIAL==MPI_THREAD_FUNNELED *****
      call MPI_Init(mpierr)
c      call MPI_Init_thread(MPI_THREAD_FUNNELED,provided,mpierr)
c      write(*,*) "MPI_init_thread, provided=",
c     >           provided,MPI_THREAD_FUNNELED

      comm_world = MPI_COMM_WORLD


      call MPI_Comm_rank(comm_world,taskid,mpierr)
      call MPI_Comm_size(comm_world,np,mpierr)

*     **** set up 3d processor grid = np x 1 x 1****
      np_i = np
      np_j = 1
      np_k = 1
      do i=0,np-1
        procNd(1+i) = i
      end do
      taskid_i = taskid
      taskid_j = 0
      taskid_k = 0
      comm_i   = comm_world
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


*     *** local variables ***
      integer i,j,icount,ierr
      integer tmp(200),tmp2(200),mpi_group

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
        procNd(1 + i + j*np_i) = icount
        icount = mod((icount+1),np)
      end do
      end do


*     **** set global processor group ****
!$OMP BARRIER
!$OMP MASTER
      call MPI_Comm_group(comm_world,mpi_group,ierr)

      do i=0,np_i-1
        tmp(1+i) = procNd(1 + i + taskid_j*np_i) 
      end do
      call MPI_Group_incl(mpi_group,np_i,tmp,group_i,ierr)
      call MPI_Comm_create(comm_world,group_i,comm_i,  ierr)

      do j=0,np_j-1
        tmp(1+j) = procNd(1 + taskid_i + j*np_i) 
      end do
      call MPI_Group_incl(mpi_group,np_j,tmp,group_j,ierr)
      call MPI_Comm_create(comm_world,group_j,comm_j,  ierr)
!$OMP END MASTER
!$OMP BARRIER


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
!$OMP BARRIER
!$OMP MASTER
      call MPI_Comm_free(comm_i,  mpierr)
      call MPI_Group_free(group_i,mpierr)
      call MPI_Comm_free(comm_j,  mpierr)
      call MPI_Group_free(group_j,mpierr)
!$OMP END MASTER
!$OMP BARRIER
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
      integer tmp(200),tmp2(200),mpi_group

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
      call MPI_COMM_group(comm_world,mpi_group,ierr)

      do i=0,np_i-1
        tmp(1+i) = procNd(1 
     >                           + i 
     >                           + taskid_j*np_i 
     >                           + taskid_k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_i,tmp,group_i,ierr)
      call MPI_Comm_create(comm_world,group_i,comm_i,ierr)

      do j=0,np_j-1
        tmp(1+j) = procNd(1 
     >                           + taskid_i 
     >                           + j*np_i
     >                           + taskid_k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_j,tmp,group_j,ierr)
      call MPI_Comm_create(comm_world,group_j,comm_j,ierr)

      do k=0,np_k-1
        tmp(1+k) = procNd(1 
     >                           + taskid_i 
     >                           + taskid_j*np_i
     >                           + k*np_i*np_j) 
      end do
      call MPI_Group_incl(mpi_group,np_k,tmp,group_k,ierr)
      call MPI_Comm_create(comm_world,group_k,comm_k,ierr)

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
!$OMP BARRIER
!$OMP MASTER
      call MPI_Comm_free(comm_i,  mpierr)
      call MPI_Group_free(group_i,mpierr)
      call MPI_Comm_free(comm_j,  mpierr)
      call MPI_Group_free(group_j,mpierr)
      call MPI_Comm_free(comm_k,  mpierr)
      call MPI_Group_free(group_k,mpierr)
!$OMP END MASTER
!$OMP BARRIER

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
!$OMP BARRIER
!$OMP MASTER
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_MAX,comm_world,mpierr)
         sum = sumall
!$OMP END MASTER
!$OMP BARRIER
      end if
      return
      end


*     ***********************************
*     *                                 *
*     *         Parallel_IMaxAll        *
*     *                                 *
*     ***********************************

      subroutine Parallel_IMaxAll(isum)
c     implicit none
      integer  isum

#include "Parallel.fh"
#include "mpif.h"

      integer msglen,mpierr
      integer isumall
      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         msglen = 1
         call MPI_Allreduce(isum,isumall,msglen,MPI_INTEGER,
     >                       MPI_MAX,comm_world,mpierr)
         isum = isumall
!$OMP END MASTER
!$OMP BARRIER
         msglen = 1
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
!$OMP BARRIER
!$OMP MASTER
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_SUM,comm_world,mpierr)
         sum = sumall
!$OMP END MASTER
!$OMP BARRIER
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
!$OMP BARRIER
!$OMP MASTER
         msglen = 1

         call MPI_Allreduce(sum,sumall,msglen,MPI_INTEGER,
     >                       MPI_SUM,comm_world,mpierr)
         sum = sumall
!$OMP END MASTER
!$OMP BARRIER
      end if
      return
      end

*     ***********************************
*     *                                 *
*     *      Parallel_Vector_SumAll     *
*     *                                 *
*     ***********************************

      subroutine Parallel_Vector_SumAll(n,sum0,sumall)
c     implicit none
      integer n
      real*8  sum0(*),sumall(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variable ****
      logical value
      integer msglen,mpierr


      call nwpw_timing_start(2)
!$OMP BARRIER
      if (np.gt.1) then
!$OMP MASTER
      msglen = n
      call MPI_Allreduce(sum0,sumall,msglen,
     >                MPI_DOUBLE_PRECISION,
     >                MPI_SUM,comm_world,mpierr)
      call dcopy(n,sumall,1,sum0,1)
!$OMP END MASTER
!$OMP BARRIER
      end if
      call nwpw_timing_end(2)
      return
      end

*     ***********************************
*     *                                 *
*     *      Parallel_Vector_SumAll2    *
*     *                                 *
*     ***********************************

      subroutine Parallel_Vector_SumAll2(n,sum0,sumall)
c     implicit none
      integer n
      real*8  sum0(*),sumall(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variable ****
      logical value
      integer msglen,mpierr

      call nwpw_timing_start(2)

!$OMP BARRIER
!$OMP MASTER
      if (np.gt.1) then
      msglen = n
      call MPI_Allreduce(sum0,sumall,msglen,
     >                MPI_DOUBLE_PRECISION,
     >                MPI_SUM,comm_world,mpierr)
      else
         call dcopy(n,sum0,1,sumall,1)
      end if
!$OMP END MASTER
!$OMP BARRIER

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
      integer sumall(200),sumall1(200)


      call nwpw_timing_start(2)

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER

      msglen = n
      call MPI_Allreduce(sum,sumall,msglen,
     >                MPI_INTEGER,
     >                MPI_SUM,comm_world,mpierr)
      call icopy(n,sumall,1,sum,1)

!$OMP END MASTER
!$OMP BARRIER
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
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(sum,1,MPI_DOUBLE_PRECISION,
     >                  psend,comm_world,ierr)
!$OMP END MASTER
!$OMP BARRIER
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

!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(sum,nsize,MPI_DOUBLE_PRECISION,
     >                  psend,comm_world,ierr)
!$OMP END MASTER
!$OMP BARRIER

      end if

      return
      end

*     ***********************************
*     *                                 *
*     *      Parallel_Brdcst_ivalue     *
*     *                                 *
*     ***********************************

      subroutine Parallel_Brdcst_ivalue(psend,isum)
      implicit none
      integer psend
      integer isum

#include "Parallel.fh"
#include "mpif.h"

      integer ierr

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(isum,1,MPI_DOUBLE_PRECISION,
     >                  psend,comm_world,ierr)
!$OMP END MASTER
!$OMP BARRIER
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
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(isum,nsize,MPI_INTEGER,
     >                  psend,comm_world,ierr)
!$OMP END MASTER
!$OMP BARRIER
      end if

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
      integer status(MPI_STATUS_SIZE*4*200),mpierr
     
      if (nreq.gt.0) then

!$OMP BARRIER
!$OMP MASTER

*     **** wait for completion of mp_send, also do a sync ****
      call MPI_WAITALL(nreq,req,status,mpierr)
c      call MPI_WAITALL(nreq,req,MPI_STATUSES_IGNORE,mpierr)

*     *** may need to check status here??? ***
!$OMP END MASTER
!$OMP BARRIER

      end if
      return
      end


      integer function getcomm(ic)
      implicit none
      integer ic

#include "Parallel.fh"
#include "mpif.h"

      integer icc
      if (ic.eq.1) then
        icc = comm_i
      else if (ic.eq.2) then
        icc = comm_j
      else if (ic.eq.3) then
        icc = comm_k
      else 
        icc = comm_world
      end if

      getcomm = icc
      return
      end

*     ***********************************
*     *                                 *
*     *         Parallela_MaxAll        *
*     *                                 *
*     ***********************************

      subroutine Parallela_MaxAll(ic,sum)
c     implicit none
      real*8  sum

#include "Parallel.fh"
#include "mpif.h"
      integer ic
      real*8 sumall
      integer msglen,mpierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_MAX,getcomm(ic),mpierr)
         sum = sumall
!$OMP END MASTER
!$OMP BARRIER
      end if
      return
      end



*     ***********************************
*     *                                 *
*     *         Parallela_SumAll        *
*     *                                 *
*     ***********************************

      subroutine Parallela_SumAll(ic,sum)
c     implicit none
      integer ic
      real*8  sum

#include "Parallel.fh"
#include "mpif.h"

      real*8 sumall

      integer msglen,mpierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         msglen = 1
         call MPI_Allreduce(sum,sumall,msglen,MPI_DOUBLE_PRECISION,
     >                       MPI_SUM,getcomm(ic),mpierr)
         sum = sumall
!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end


*     ***********************************
*     *                                 *
*     *         Parallela_ISumAll       *
*     *                                 *
*     ***********************************

      subroutine Parallela_ISumAll(ic,sum)
c     implicit none
      integer ic,sum


#include "Parallel.fh"
#include "mpif.h"

      integer msglen,mpierr
      integer sumall
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         msglen = 1

         call MPI_Allreduce(sum,sumall,msglen,MPI_INTEGER,
     >                       MPI_SUM,getcomm(ic),mpierr)
         sum = sumall

!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end

*     ***********************************
*     *                                 *
*     *      Parallela_Vector_SumAll    *
*     *                                 *
*     ***********************************

      subroutine Parallela_Vector_SumAll(ic,n,sum,sumall)
c     implicit none
      integer ic,n
      real*8  sum(*),sumall(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variable ****
      logical value
      integer msglen,mpierr

*     **** temporary workspace ****

      integer  getcomm
      external getcomm


      call nwpw_timing_start(2)
      if (np.gt.1) then

!$OMP BARRIER
!$OMP MASTER

      msglen = n
      call MPI_Allreduce(sum,sumall,msglen,
     >                MPI_DOUBLE_PRECISION,
     >                MPI_SUM,getcomm(ic),mpierr)

      call dcopy(n,sumall,1,sum,1)

!$OMP END MASTER
!$OMP BARRIER
      end if
      call nwpw_timing_end(2)
      return
      end


*     ***********************************
*     *                                 *
*     *      Parallela_Vector_ISumAll   *
*     *                                 *
*     ***********************************

      subroutine Parallela_Vector_ISumAll(ic,n,sum)
c     implicit none
      integer ic,n
      integer  sum(*)

#include "Parallel.fh"
#include "mpif.h"

      logical value
      integer msglen,mpierr

*     **** temporary workspace ****
      integer sumall(200),sumall1(200)

      integer  getcomm
      external getcomm

      call nwpw_timing_start(2)

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER

      msglen = n
      call MPI_Allreduce(sum,sumall,msglen,
     >                MPI_INTEGER,
     >                MPI_SUM,getcomm(ic),mpierr)
      call icopy(n,sumall,1,sum,1)

!$OMP END MASTER
!$OMP BARRIER

      end if

      call nwpw_timing_end(2)
      return
      end





*     ***********************************
*     *                                 *
*     *      Parallela_Brdcst_value     *
*     *                                 *
*     ***********************************

      subroutine Parallela_Brdcst_value(ic,psend,sum)
      implicit none
      integer ic,psend
      real*8  sum

#include "Parallel.fh"
#include "mpif.h"

      integer ierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(sum,1,MPI_DOUBLE_PRECISION,
     >                  psend,getcomm(ic),ierr)
!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end




*     ***********************************
*     *                                 *
*     *      Parallela_Brdcst_values     *
*     *                                 *
*     ***********************************

      subroutine Parallela_Brdcst_values(ic,psend,nsize,sum)
      implicit none
      integer ic,psend,nsize
      real*8  sum(*)

#include "Parallel.fh"
#include "mpif.h"

      integer ierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(sum,nsize,MPI_DOUBLE_PRECISION,
     >                  psend,getcomm(ic),ierr)
!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end



*     ***********************************
*     *                                 *
*     *      Parallela_Brdcst_ivalue    *
*     *                                 *
*     ***********************************

      subroutine Parallela_Brdcst_ivalue(ic,psend,isum)
      implicit none
      integer ic,psend
      integer isum

#include "Parallel.fh"
#include "mpif.h"

      integer ierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(isum,1,MPI_DOUBLE_PRECISION,
     >                  psend,getcomm(ic),ierr)
!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end



*     ***********************************
*     *                                 *
*     *      Parallela_Brdcst_ivalues   *
*     *                                 *
*     ***********************************

      subroutine Parallela_Brdcst_ivalues(ic,psend,nsize,isum)
      implicit none
      integer ic,psend,nsize
      integer isum(*)

#include "Parallel.fh"
#include "mpif.h"

      integer ierr
      integer  getcomm
      external getcomm

      if (np.gt.1) then
!$OMP BARRIER
!$OMP MASTER
         call MPI_Bcast(isum,nsize,MPI_INTEGER,
     >                  psend,getcomm(ic),ierr)
!$OMP END MASTER
!$OMP BARRIER
      end if

      return
      end





*     ***********************************
*     *                                 *
*     *      Parallela_start_rotate     *
*     *                                 *
*     ***********************************

      subroutine Parallela_start_rotate(ic,shift,msgtype0,
     >                                  A1,nsize1,
     >                                  A2,nsize2,request)
      implicit none
      integer ic,shift,msgtype0
      real*8  A1(*)
      integer nsize1
      real*8  A2(*)
      integer nsize2
      integer request(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variables ****
      integer mynp,mytaskid,msglen,msgtype,mpierr,proc_to,proc_from

*     ***** external functions ****
      integer  getcomm
      external getcomm

      if (ic.eq.1) then
         mynp     = np_i
         mytaskid = taskid_i
      else if (ic.eq.2) then
         mynp     = np_j
         mytaskid = taskid_j
      else if (ic.eq.3) then
         mynp     = np_k
         mytaskid = taskid_k
      else
         mynp     = np
         mytaskid = taskid
      end if
      proc_to   = mod(mytaskid+shift+mynp,mynp)
      proc_from = mod(mytaskid-shift+mynp,mynp)

         if (nsize2.gt.0) then
         msglen  = nsize2
         msgtype = msgtype0
         call MPI_IRECV(A2,msglen,MPI_DOUBLE_PRECISION,
     >                  proc_from,
     >                  msgtype,getcomm(ic),
     >                  request(1),mpierr)
            request(3) = 1
         else
            request(3) = 0
         end if

         if (nsize1.gt.0) then
         msglen  = nsize1
         msgtype = msgtype0
         call MPI_ISEND(A1,msglen,MPI_DOUBLE_PRECISION,
     >                  proc_to,
     >                  msgtype,getcomm(ic),
     >                  request(2),mpierr)
            request(4) = 1
         else
            request(4) = 0
         end if

      if ((request(3).eq.1).and.(request(4).eq.1)) then
         request(3) = 1
      else if (request(3).eq.1) then
         request(3) = 2
      else if (request(4).eq.1) then
         request(3) = 3
      else
         request(3) = 4
      end if

      return
      end



*     ***********************************
*     *                                 *
*     *      Parallela_start_Irotate     *
*     *                                 *
*     ***********************************

      subroutine Parallela_start_Irotate(ic,shift,msgtype0,
     >                                  IA1,nsize1,
     >                                  IA2,nsize2,request)
      implicit none
      integer ic,shift,msgtype0
      integer IA1(*)
      integer nsize1
      integer IA2(*)
      integer nsize2
      integer request(*)

#include "Parallel.fh"
#include "mpif.h"

*     **** local variables ****
      integer mynp,mytaskid,msglen,msgtype,mpierr,proc_to,proc_from

*     ***** external functions ****
      integer  getcomm
      external getcomm

      if (ic.eq.1) then
         mynp     = np_i
         mytaskid = taskid_i
      else if (ic.eq.2) then
         mynp     = np_j
         mytaskid = taskid_j
      else if (ic.eq.3) then
         mynp     = np_k
         mytaskid = taskid_k
      else
         mynp     = np
         mytaskid = taskid
      end if
      proc_to   = mod(mytaskid+shift+mynp,mynp)
      proc_from = mod(mytaskid-shift+mynp,mynp)

         if (nsize2.gt.0) then
         msglen  = nsize2
         msgtype = msgtype0
         call MPI_IRECV(IA2,msglen,MPI_INTEGER,
     >                  proc_from,
     >                  msgtype,getcomm(ic),
     >                  request(1),mpierr)
            request(3) = 1
         else
            request(3) = 0
         end if

         if (nsize1.gt.0) then
         msglen  = nsize1
         msgtype = msgtype0
         call MPI_ISEND(IA1,msglen,MPI_INTEGER,
     >                  proc_to,
     >                  msgtype,getcomm(ic),
     >                  request(2),mpierr)
            request(4) = 1
         else
            request(4) = 0
         end if

      if ((request(3).eq.1).and.(request(4).eq.1)) then
         request(3) = 1
      else if (request(3).eq.1) then
         request(3) = 2
      else if (request(4).eq.1) then
         request(3) = 3
      else
         request(3) = 4
      end if

      return
      end



*     ***********************************
*     *                                 *
*     *      Parallela_end_rotate       *
*     *                                 *
*     ***********************************

      subroutine Parallela_end_rotate(request)
      implicit none
      integer request(*)

*     **** wait for completion of mp_send, also do a sync ****
      if (request(3).eq.1) then
         call Parallel_mpiWaitAll(2,request)
      else if (request(3).eq.2) then
         call Parallel_mpiWaitAll(1,request)
      else if (request(3).eq.3) then
         call Parallel_mpiWaitAll(1,request(2))
      endif

      return
      end



*     ***********************************
*     *                                 *
*     *      Parallel_send_values       *
*     *                                 *
*     ***********************************
      subroutine Parallel_send_values(pto,msgtype,nsize,rval)
      implicit none
      integer pto,msgtype,nsize
      real*8 rval(*)

#include "Parallel.fh"
#include "mpif.h"
      integer mpierr

      call MPI_SEND(rval,nsize,MPI_DOUBLE_PRECISION,
     >              pto,msgtype,comm_world,mpierr)
      return
      end 

*     ***********************************
*     *                                 *
*     *      Parallel_send_ivalues      *
*     *                                 *
*     ***********************************
      subroutine Parallel_send_ivalues(pto,msgtype,nsize,ival)
      implicit none
      integer pto,msgtype,nsize
      integer ival(*)

#include "Parallel.fh"
#include "mpif.h"
      integer mpierr

      call MPI_SEND(ival,nsize,MPI_INTEGER,
     >              pto,msgtype,comm_world,mpierr)
      return
      end




*     ***********************************
*     *                                 *
*     *      Parallel_send_characters   *
*     *                                 *
*     ***********************************
      subroutine Parallel_send_characters(pto,msgtype,nsize,cval)
      implicit none
      integer pto,msgtype,nsize
      character cval(*)

#include "Parallel.fh"
#include "mpif.h"
      integer mpierr

      call MPI_SEND(cval,nsize,MPI_CHARACTER,
     >              pto,msgtype,comm_world,mpierr)
      return
      end



*     ***********************************
*     *                                 *
*     *      Parallel_recv_values       *
*     *                                 *
*     ***********************************
      subroutine Parallel_recv_values(pfrom,msgtype,nsize,rval)
      implicit none
      integer pfrom,msgtype,nsize
      real*8 rval(*)

#include "Parallel.fh"
#include "mpif.h"

      logical value
      integer status(MPI_STATUS_SIZE*4),mpierr

*     **** allocate status memory ****
      call MPI_RECV(rval,nsize,MPI_DOUBLE_PRECISION,
     >              pfrom,msgtype,comm_world,status,
     >              mpierr)

      return
      end

*     ***********************************
*     *                                 *
*     *      Parallel_recv_ivalues       *
*     *                                 *
*     ***********************************
      subroutine Parallel_recv_ivalues(pfrom,msgtype,nsize,ival)
      implicit none
      integer pfrom,msgtype,nsize
      integer ival(*)

#include "Parallel.fh"
#include "mpif.h"

      logical value
      integer status(MPI_STATUS_SIZE*4),mpierr

*     **** allocate status memory ****

*     ***** allocate temporary space ****
      call MPI_RECV(ival,nsize,MPI_INTEGER,
     >              pfrom,msgtype,comm_world,status,mpierr)

      return
      end




*     ***********************************
*     *                                 *
*     *      Parallel_recv_characters   *
*     *                                 *
*     ***********************************
      subroutine Parallel_recv_characters(pfrom,msgtype,nsize,cval)
      implicit none
      integer  pfrom,msgtype,nsize
      character cval(*)

#include "Parallel.fh"
#include "mpif.h"


      logical value
      integer status(MPI_STATUS_SIZE*4),mpierr

*     **** allocate status memory ****

      call MPI_RECV(cval,nsize,MPI_CHARACTER,
     >              pfrom,msgtype,comm_world,status,
     >              mpierr)

      return
      end


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


*
* $Id: Parallel.F 26777 2015-01-30 18:59:15Z mjacquelin $
*

* Parallel.f
* Author - Eric Bylaska
*
*   These routines are to be used to keep track of the parallel message
* passing variables, as well as iniitialize and deinitialize the
* message passing routines.
*



*     *************************************
*     *                                   *
*     *        Parallel_Finalize          *
*     *                                   *
*     *************************************

      subroutine Parallel_Finalize()
      implicit none

#include "Parallel.fh"

      integer mpierr
      call MPI_Finalize(mpierr)

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

*     *************************************
*     *                                   *
*     *         Parallel_comm_world       *
*     *                                   *
*     *************************************
      integer function Parallel_comm_world()
      implicit none

#include "Parallel.fh"

      Parallel_comm_world = comm_world
      return
      end





*     *************************************
*     *                                   *
*     *        Parallel_thread_init       *
*     *                                   *
*     *************************************
      subroutine Parallel_thread_init()
      implicit none
#include "Parallel.fh"

#ifdef USE_OPENMP
      integer  omp_get_thread_num,omp_get_num_threads
      external omp_get_thread_num,omp_get_num_threads
#endif

#ifdef USE_OPENMP
      threadid = omp_get_thread_num()
      nthreads = omp_get_num_threads()
#else
      threadid = 0
      nthreads = 1
#endif

      !write(*,*) "---threadid=",threadid
      return
      end

      integer function Parallel_threadid()
      implicit none
#include "Parallel.fh"
#ifdef USE_OPENMP
      integer  omp_get_thread_num
      external omp_get_thread_num
#endif

#ifdef USE_OPENMP
      Parallel_threadid = omp_get_thread_num()
#else
      Parallel_threadid = 0
#endif
      return
      end

      integer function Parallel_nthreads()
      implicit none
#include "Parallel.fh"
#ifdef USE_OPENMP
      integer  omp_get_num_threads
      external omp_get_num_threads
#endif

#ifdef USE_OPENMP
      Parallel_nthreads = omp_get_num_threads()
#else
      Parallel_nthreads = 1
#endif

      return
      end

      integer function Parallel_maxthreads()
      implicit none
#include "Parallel.fh"
#ifdef USE_OPENMP
      integer  omp_get_max_threads
      external omp_get_max_threads
#endif

#ifdef USE_OPENMP
      Parallel_maxthreads = omp_get_max_threads()
#else
      Parallel_maxthreads = 1
#endif

      return
      end





      subroutine  icopy(n,dx,incx,dy,incy)
c
c     copies a vector, x, to a vector, y.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      INTEGER          dx(1),dy(1)
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,7)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dx(i)
   30 continue
      if( n .lt. 7 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,7
        dy(i) = dx(i)
        dy(i + 1) = dx(i + 1)
        dy(i + 2) = dx(i + 2)
        dy(i + 3) = dx(i + 3)
        dy(i + 4) = dx(i + 4)
        dy(i + 5) = dx(i + 5)
        dy(i + 6) = dx(i + 6)
   50 continue
      return
      end

*     *************************************
*     *                                   *
*     *        Parallel_index_1dblock     *
*     *                                   *
*     *************************************

      integer function Parallel_index_1dblock(m,mb,i)
      implicit none
      integer m,mb,i
      integer ms,r

      if (i.ge.mb) then
         ms = m
      else
         ms = (m/mb)*i
         r = mod(m,mb)
         if (i.lt.r) then
            ms = ms + i
         else
            ms = ms + r
         end if
      end if
      Parallel_index_1dblock = ms
      return
      end


*     *******************************************
*     *                                         *
*     *        Parallel_matrixblocking          *
*     *                                         *
*     *******************************************
*
*   This routine computes mp,np such that
*   mp*np = nthr and min |(mp/np)-(m/n)|
*
*   The justification for trying to keep the ratio of mp/np ~= m/n is that having
*  the subblocks mimimic the overall matrix will produce distribution with blocks are 
*  relatively equal in size.
*
      subroutine Parallel_matrixblocking(nthr,m,n,mb,nb)
      implicit none
      integer nthr,m,n,mb,nb

*     **** local variables ****
      integer ii,jj,mm,nn
      real*8 ratio

      ratio = dble(m)/dble(n)
      mb = nthr
      nb = 1
      do nn =1,nthr
         ii = (nthr-1)/nn - 1
         jj = (nthr+1)/nn + 1
         if (ii.lt.1) ii = 1
         if (jj.gt.nthr) jj=nthr
         do mm=ii,jj
            if ((nn*mm).eq.nthr) then
               if (dabs(dble(mm)/dble(nn) - ratio) .lt.
     >             dabs(dble(mb)/dble(nb) - ratio)) then
                  mb = mm
                  nb = nn
               end if
            end if
         end do
      end do
      return
      end

