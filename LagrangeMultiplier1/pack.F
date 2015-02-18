      subroutine epack_cc_dot(nida,nidb,A,B,w)
      implicit none
      integer nida,nidb
      complex*16 A(*),B(*)
      real*8 w

      real*8   ddot
      external ddot

      w = ddot(2*nida,A,1,B,1)
      w = w + 2*ddot(2*nidb,A(nida+1),1,B(nida+1),1)
      call Parallela_SumAll(4,w)
      return
      end
 
