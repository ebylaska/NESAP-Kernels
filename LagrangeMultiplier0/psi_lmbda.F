*     ********************************************
*     *                                          *
*     *                psi_lmbda                 *
*     *                                          *
*     ********************************************

      subroutine psi_lmbda(ispin,ne,nemaxq,nida,nidb,
     >                     psi1,psi2,
     >                     dte,
     >                     lmbda,tmp,ierr)

      implicit none
      integer ispin,ne(2),nemaxq,nida,nidb
      complex*16 psi1(nida+nidb,nemaxq)
      complex*16 psi2(nida+nidb,nemaxq)
      real*8     dte
      real*8     lmbda(*)
      real*8     tmp(*)
      integer    ierr

      integer MASTER,taskid
      parameter (MASTER=0)

*     **** local variables ****
      logical notgram
      integer ms,it
      integer nn,i
      integer s11,s12,s21,s22,st1,st2,sa1,sa0
      real*8  adiff

*     ::::  iteration limit and tolerence for non-liner equations  ::::
      integer itrlmd
      real*8  convg
      parameter (itrlmd=120, convg=1.0d-15)

*     **** external functions ****
      real*8   eDneall_m_dmax
      external eDneall_m_dmax

      call nwpw_timing_start(3)


*::::::::::::::::::::::  Lagrangian multipliers  ::::::::::::::::::::::
      DO 640 ms=1,ispin
        notgram=.true.
        IF(ne(ms).le.0) GO TO 640

        !call Dneall_m_size(ms,nn)
        nn = ne(ms)*ne(ms)
        s11  = 0*nn + 1
        s21  = 1*nn + 1
        s22  = 2*nn + 1
        s12  = 3*nn + 1
        sa0  = 4*nn + 1
        sa1  = 5*nn + 1
        st1  = 6*nn + 1
        st2  = 7*nn + 1
        call eDneall_ffm_sym_Multiply(ms,psi2,psi2,
     >                                nida,nidb,ispin,ne,
     >                                tmp(s22),tmp(sa0))
        call eDneall_ffm_sym_Multiply(ms,psi2,psi1,
     >                                nida,nidb,ispin,ne,
     >                                tmp(s21),tmp(sa1))
        call eDneall_ffm_sym_Multiply(ms,psi1,psi1,
     >                                nida,nidb,ispin,ne,
     >                                tmp(s11),tmp(st1))

*       ***** scale the overlap matrices ****
        call eDneall_m_scale_s22(ms,ispin,ne,dte,tmp(s22))
        call eDneall_m_scale_s21(ms,ispin,ne,dte,tmp(s21))
        call eDneall_m_scale_s11(ms,ispin,ne,dte,tmp(s11))

        call dcopy(nn,tmp(s21),1,tmp(s12),1)
        call dcopy(nn,tmp(s22),1,tmp(sa0),1)

        do it=1,itrlmd
          call dcopy(nn,tmp(s22),1,tmp(sa1),1)

          call eDneall_mmm_Multiply(ms,ispin,ne,
     >                              tmp(s21),tmp(sa0),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))
          call eDneall_mmm_Multiply(ms,ispin,ne,
     >                              tmp(sa0),tmp(s12),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))
          call eDneall_mmm_Multiply(ms,ispin,ne,
     >                              tmp(s11),tmp(sa0),1.0d0,
     >                              tmp(st1),0.0d0,tmp(st2))
          call eDneall_mmm_Multiply(ms,ispin,ne,
     >                              tmp(sa0),tmp(st1),1.0d0,
     >                              tmp(sa1),1.0d0,tmp(st2))

          call dcopy(nn,tmp(sa1),1,tmp(st1),1)
          call daxpy(nn,(-1.0d0),tmp(sa0),1,tmp(st1),1)

          adiff = eDneall_m_dmax(ms,ispin,ne,tmp(st1))
          if(adiff.lt.convg) GO TO 630
          if (adiff.gt.1.0d10) go to 620
          call dcopy(nn,tmp(sa1),1,tmp(sa0),1)
        end do

  620   continue
        ierr=10 
        call Parallel_taskid(taskid)
        if (taskid.eq.MASTER) then
          write(6,*) 
     >     'Warning: Lagrange Multiplier tolerance too high:',adiff
          write(6,*) '        +Try using a smaller time step'
          write(6,*) '        +Gram-Schmidt being performed, spin:',ms
        end if
c        call Dneall_f_GramSchmidt(ms,psi2,npack1)
        notgram = .false.

  630   continue

*       :::::::::::::::::  correction due to the constraint  :::::::::::::::::
        if (notgram)
     >     call eDneall_fmf_Multiply(ms,ispin,ne,
     >                              psi1,nida+nidb,
     >                              tmp(sa1), dte,
     >                              psi2,1.0d0)

        call eDneall_mm_Expand(ms,ne,tmp(sa1),lmbda)
  640 continue

c*:::::::::::::::::  correction due to the constraint  :::::::::::::::::
      call nwpw_timing_end(3)

      return
      end


      subroutine eDneall_ffm_sym_Multiply(mb,A1,A2,
     >                                    nida,nidb,ispin,ne,hml,tmp)
      implicit none
      integer mb
      complex*16 A1(*),A2(*)
      integer nida,nidb,ispin,ne(2)
      real*8 hml(*),tmp(*)

      integer ms,ms1,ms2,ishift2,n,shift,shift2,npack1

      call nwpw_timing_start(15)    
      npack1 = nida+nidb
      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
         shift  = 1 + (ms-1)*ne(1)*npack1
         shift2 = 1 + (ms-1)*ishift2
         n     = ne(ms)
         if (n.le.0) go to 30

         call epack_ccm_sym_dot(nida,nidb,n,
     >                     A1(shift),
     >                     A2(shift),
     >                     hml(shift2),tmp)
  30     continue
      end do

      call nwpw_timing_end(15)    
      return
      end



*     ***********************************
*     *                                 *
*     *         epack_ccm_sym_dot       *       
*     *                                 *
*     ***********************************

      subroutine epack_ccm_sym_dot(nida,nidb,n,A,B,matrix,tmp)
      implicit none
      integer    nida,nidb,n
      complex*16 A(*)
      complex*16 B(*)
      real*8     matrix(n,n)
      real*8     tmp(*)

*     **** local variables ****
      integer j,k
      integer np,npack,npack2


      call nwpw_timing_start(2)
      call Parallel_np(np)

      npack  = (nida+nidb)
      npack2 = 2*npack


      do k=1,n
        call DGEMM('T','N',k,1,2*nida,
     >             1.0d0,
     >             A,npack2,
     >             B(1+(k-1)*npack),npack2,
     >             0.0d0,
     >             matrix(1,k),k)
        call DGEMM('T','N',k,1,npack2-2*nida,
     >             2.0d0,
     >             A(1+nida),npack2,
     >             B(1+nida+(k-1)*npack),npack2,
     >             1.0d0,
     >             matrix(1,k),k)
      end do

      do k=1,n
      do j=k+1,n
        matrix(j,k) = matrix(k,j)
      end do
      end do

      if (np.gt.1) call Parallel_Vector_SumAll(n*n,matrix,tmp)


      call nwpw_timing_end(2)
      return
      end



    
c     ****************************************
c     *                                      *
c     *        Dneall_m_scale_s22           *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s22(mb,ispin,ne,dte,s22)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s22(*)
        

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt
      

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s22(indx) = (1.0d0 - s22(indx))*0.5d0/dte

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s22(indx)  = -s22(indx)*0.5d0/dte
              s22(indxt) = s22(indx)
           end do
        end do

 30     continue
      end do

      return
      end



c     ****************************************
c     *                                      *
c     *        Dneall_m_scale_s21            *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s21(mb,ispin,ne,dte,s21)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s21(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt


      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s21(indx) = (1.0d0 - s21(indx))*0.5d0

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s21(indx)  = -s21(indx)*0.5d0
              s21(indxt) = s21(indx)
           end do
        end do

 30     continue
      end do
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_m_scale_s11           *
c     *                                      *
c     ****************************************

      subroutine eDneall_m_scale_s11(mb,ispin,ne,dte,s11)
      implicit none
      integer mb,ispin,ne(2)
      real*8 dte
      real*8 s11(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2,k,j,indx,indxt
      integer tid,nthr


      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = (ms-1)*ishift2

        do k=1,ne(ms)
           indx = k + (k-1)*ne(ms) + shift2
           s11(indx) = -s11(indx)*0.5d0*dte

           do j=k+1,ne(ms)
              indx  = j + (k-1)*ne(ms) + shift2
              indxt = k + (j-1)*ne(ms) + shift2

              s11(indx)  = -s11(indx)*0.5d0*dte
              s11(indxt) = s11(indx)
           end do
        end do

 30     continue
      end do
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_mmm_Multiply          *
c     *                                      *
c     ****************************************

      subroutine eDneall_mmm_Multiply(mb,ispin,ne,A,B,alpha,C,beta,tmp)
      implicit none
      integer mb,ispin,ne(2)
      real*8 A(*),B(*),C(*)
      real*8 alpha,beta
      real*8 tmp(*)

*     **** local variables ****
      integer MASTER,taskid,np
      parameter (MASTER=0)
      integer ms,ms1,ms2,n,shift2,ishift2
      integer i,j,pindx
      integer ishiftA,ishiftB,ishiftC


      call Parallel_taskid(taskid)
      call Parallel_np(np)

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
         n     = ne(ms)
         if (n.le.0) go to 30
         shift2 = 1 + (ms-1)*ishift2

         !*** completely serial ****
         call DGEMM('N','N',n,n,n,
     >             alpha,
     >             A(shift2), n,
     >             B(shift2), n,
     >             beta,
     >             C(shift2), n)

  30     continue
      end do
      if (mb.eq.0) then
         call Parallel_Brdcst_values(MASTER,
     >                   ne(1)*ne(1)+ne(2)*ne(2),C)
      else
         call Parallel_Brdcst_values(MASTER,ne(mb)*ne(mb),C)
      end if


      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_m_dmax                 *
c     *                                      *
c     ****************************************
         
      double precision function eDneall_m_dmax(mb,ispin,ne,A)
      implicit none
      integer mb,ispin,ne(2)
      real*8 A(*)

*     **** local variables ****
      integer ms,ms1,ms2,shift2,ishift2
      double precision adiff1, adiff2
           
      integer  idamax
      external idamax

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      adiff1 = 0.0d0
      adiff2 = 0.0d0
      do ms=ms1,ms2
        if (ne(ms).le.0) go to 30
        shift2 = 1 + (ms-1)*ishift2

        adiff1 = adiff2
        adiff2 = A(shift2-1+idamax(ne(ms)*ne(ms),A(shift2),1))
        adiff2 = dabs(adiff2)
        if (adiff2.gt.adiff1) adiff1 = adiff2
 30     continue
      end do

      eDneall_m_dmax = adiff1
      return
      end


c     ****************************************
c     *                                      *
c     *        eDneall_fmf_Multiply          *
c     *                                      *
c     ****************************************
          
      subroutine eDneall_fmf_Multiply(mb,ispin,ne,Ain,npack1,
     >                                hml,alpha,
     >                                Aout,beta)
      implicit none
      integer    mb,ispin,ne(2)
      complex*16 Ain(*)
      integer    npack1
      real*8     hml(*)
      real*8     alpha
      complex*16 Aout(*)
      real*8     beta
        

*     **** local variables ****
      integer tid,nthr,mstart,mend
      integer ms,ms1,ms2,n,shift,shift2,shift3,ishift2,ishift3


      call nwpw_timing_start(16)    

      if (mb.eq.0) then
         ms1 = 1
         ms2 = ispin
         ishift2 = ne(1)*ne(1)
      else
         ms1 = mb
         ms2 = mb
         ishift2 = 0
      end if

      do ms=ms1,ms2
         n     = ne(ms)
         if (n.le.0) go to 30
         shift  = 1 + (ms-1)*ne(1)*npack1
         shift2 = 1 + (ms-1)*ishift2
      
         call DGEMM('N','N',2*npack1,n,n,
     >             (alpha),
     >             Ain(shift), 2*npack1,
     >             hml(shift2),    n,
     >             (beta),
     >             Aout(shift),2*npack1)
   30    continue
      end do

      call nwpw_timing_end(16)
      return
      end

c     ****************************************
c     *                                      *
c     *        eDneall_mm_Expand             *
c     *                                      *
c     ****************************************

      subroutine eDneall_mm_Expand(mb,ne,A,A0)
      implicit none
      integer mb,ne(2)
      real*8 A(*),A0(*)


*     **** local variables ****
      integer shift2,nn

      shift2 = 1
      if (mb.eq.0) then
         nn     = ne(1)*ne(1) + ne(2)*ne(2)
         shift2 = 1
      else if (mb.eq.1) then
         nn     = ne(1)*ne(1)
         shift2 = 1
      else if (mb.eq.2) then
         nn     = ne(2)*ne(2)
         shift2 = 1+ne(1)*ne(1)
      end if


      call dcopy(nn,A,1,A0(shift2),1)
      return
      end


