



      subroutine  dcopy(n,dx,incx,dy,incy)
      implicit none
      integer n,incx,incy
      real*8 dx(*)
      real*8 dy(*)
c
c     copies a vector, x, to a vector, y.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      integer i,ix,iy,m,mp1
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


      real*8 function ddot(n,dx,incx,dy,incy)
      implicit none
      integer n,incx,incy
      real*8 dx(*),dy(*)
      integer ix,iy,i
      real*8 tsum
      tsum = 0.0d0
      ix = 1
      iy = 1
      do i=1,n
         tsum = tsum + dx(ix)*dy(iy)
         ix = ix + incx
         iy = iy + incy
      end do

      ddot = tsum
      return
      end



      subroutine dscal(n,alpha,A,ishftA)
      implicit none
      integer n,ishftA
      real*8 alpha,A(*)

      integer i,ia

      ia = 1
      do i=1,n
         A(ia) = alpha*A(ia)
         ia = ia + ishftA
      end do
      return
      end
