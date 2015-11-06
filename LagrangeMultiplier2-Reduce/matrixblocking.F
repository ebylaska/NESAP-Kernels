
*     *******************************************
*     *                                         *
*     *        Parallel_matrixblock             *
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
      subroutine Parallel_matrixblocks(nthr,m,n,mb,nb)
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


      integer function Parallel_1dblocksize(m,mb,b)
      implicit none
      integer m,mb,b

*     **** local variables ****
      integer i,bsz

      bsz = 0
      do i=1,m
        ib= mod(i-1,nthrnp) + 1
        if (ib.eq.b) bsz = bsz + 1
      end do
      Parallel_1dblocksize = bsz
      return
      end 

*     *******************************************
*     *                                         *
*     *        Parallel_matrixblocking          *
*     *                                         *
*     *******************************************
*
      subroutine Parallel_matrixblocking(nthr,taskid,np,m,n,
     >                                   algorithm,
     >                                   mstart,nstart,
     >                                   mblock,nblock)
      implicit none
      integer nthr,taskid,np,m,n
      integer algorithm
      integer mstart(nthr),nstart(nthr)
      integer mblock(nthr),nblock(nthr)

c     **** local variables ****
      integer mb,nb

      call Parallel_matrixblocks(nthr*np,m,n,mb,nb)
      do i=1,nthr
         mstart(i) = 0
         nstart(i) = 0
         mblock(i) = 0
         nblock(i) = 0
      end do
      tid + taskid*nthr
      mp
      0 1 2 3  0 1 2 3   0 1 2 3
      0 1 2 3  4 5 6 7   8 9 10 11
      do itaskid=0,np-1
         do itid = 1,nthr
            b = itid + itaskid*nthr
      do i=1,m
        itt= mod(i-1,nthr*np)
     

      do i=1,m
        it  = mod(i-1,nthr)
        itt = mod(i-1,nthr*np)
        itaskid = (itt-it)/nthr
        if (itaskid.eq.taskid) mblock(it+1) = mblock(it+1) + 1
      end do
      count = 1
      do i=1,mb
         mstart(i) = count
         count = count + mblock(i)
      end do
