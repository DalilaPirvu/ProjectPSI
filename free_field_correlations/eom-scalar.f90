#include "macros.h"
#include "fldind.h"
#define FIELD_TYPE Field_Model

module eom
  use fftw3
  use constants
  implicit none

!  integer, parameter :: nLat = 2048*2, nVar = 2*nLat+1, nyq = nLat/2+1
  integer, parameter :: nLat = 4096, nVar = 2*nLat+1, nyq = nLat/2+1
  real(dl), dimension(1:nVar), target :: yvec
  real(dl), parameter :: nu = 2.e-3 ! BEC parameter
  real(dl), parameter :: omega = 0.25*50._dl*2._dl*nu**0.5 ! omega = freq of osc potential
  real(dl), parameter :: del = (nu/2._dl)**0.5*(1._dl + 0.4_dl) ! del = height of barrier
  real(dl), parameter :: len = 0.5*50._dl / (2.*nu)**0.5  ! len = side length of box
  real(dl), parameter :: dx = len/dble(nLat), dk = twopi/len ! dx = size of one site, dk = fundamental mode, step in Fourier space
  real(dl), parameter :: rho = 2.**(-3)*200._dl*2._dl*nu**0.5 !
  real(dl), parameter :: lambda = del*(2._dl/nu)**0.5 ! if >1, potential has false min
  real(dl), parameter :: m2eff = 4._dl*nu*(-1._dl+lambda**2) ! effective mass squared
!  real(dl), parameter :: m2eff = 1._dl ! effective mass squared
  type(transformPair1D) :: tPair
contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  subroutine derivs(yc,yp) ! equations of motion
    real(dl), dimension(:), intent(in) :: yc
    real(dl), dimension(:), intent(out) :: yp
    yp(TIND) = 1._dl ! Uncomment to track time as a variable
    yp(FLD) = yc(DFLD)
    yp(DFLD) = - m2eff * yc(FLD)
    tPair%realSpace(:) = yc(FLD)
    call laplacian_1d_wtype(tPair,dk)
    yp(DFLD) = yp(DFLD) + tPair%realSpace(:)
  end subroutine derivs
end module eom
