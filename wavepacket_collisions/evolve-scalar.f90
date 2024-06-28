!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!>COLLISIONS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#include "macros.h"
#include "fldind.h"
#define SMOOTH 1

program Gross_Pitaevskii_1d
  ! the following modules are incorporated:
  use, intrinsic :: iso_c_binding
  use gaussianRandomField
  use integrator
  use constants
  use eom
  implicit none
  
  real(dl), dimension(:,:), pointer :: fld
  real(dl), pointer :: time
  real(dl) :: alph = 8._dl, sigma = 0.15_dl/sqrt(m2eff)
  integer, parameter :: inFile = 70, cpFile = 71 ! parameter type means constant
  integer :: n_cross = 2, nTime = 32 ! n_cross = number of bubble intersections
  real(dl) :: width, amplitude, width0 = 1._dl, amplitude0 = 0._dl
  integer :: lag, mean1, mean2, sim, wid, amp, nWid = 80, nAmp = 80+25

  fld(1:nLat,1:2) => yvec(1:nVar-1) ! store the field in yvec
  time => yvec(nVar) ! last position stores the time?
  call initialize_rand(97786856,12)
  call setup(nVar)

  mean1 = int(nLat / 4._dl + nLat / 10._dl)
  mean2 = nLat-mean1

  do lag = 1, 2

    do wid = 1, nWid
      width = width0 + real(wid) * 0.05_dl / sqrt(m2eff)

      do amp = 1, nAmp
        amplitude = amplitude0 + twopi / 2._dl * real(amp) / 200._dl

        do sim = 1, 20 ! for fluctuation
          call initialise_fields(fld, nyq, 0.25*twopi, nyq/2, mean1, mean2, width, amplitude, lag)
          call time_evolve(dx/alph, 4*nLat*n_cross, width, amplitude, lag, sim)

          print*, " Simulation done!"
        enddo
      enddo
    enddo
  enddo

  print*, 'm2eff = ', m2eff, ' len = ', len, 'dt = ', dx/alph, 'dx = ', dx, 'dk = ', dk, 'sigma = ', sigma

contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine initialise_fields(fld,kmax,phi,klat, mean1, mean2, width, amplitude, lag)
    real(dl), dimension(:,:), intent(inout) :: fld
    integer, intent(in) :: kmax
    real(dl), intent(in), optional :: phi
    integer, intent(in), optional :: klat, mean1, mean2, lag
    real(dl) :: amplitude, width
    integer :: kc
    real(dl) :: phiL

    kc = nLat/2+1; if (present(klat)) kc = klat
    phiL = 0.5_dl*twopi; if (present(phi)) phiL = phi
    call initialise_mean_fields(fld)
    yvec(2*nLat+1) = 0._dl ! Add a tcur pointer here

    call initialize_vacuum_fluctuations(fld,len,m2eff,kmax,phiL,kc) ! Change this call as necessary

    call initialize_wavepacket(fld, mean1, width, amplitude, 1._dl, lag) ! direction +1 means move right
    call initialize_wavepacket(fld, mean2, width, amplitude, -1._dl, lag)

  end subroutine initialise_fields
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine initialise_mean_fields(fld)
    real(dl), dimension(:,:), intent(out) :: fld
    fld(:,1) = 0.5_dl*twopi
    fld(:,2) = 0._dl
  end subroutine initialise_mean_fields
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine initialize_wavepacket(fld, mean, width, amplitude, direction, lag)
    real(dl), dimension(:,:), intent(inout) :: fld
    real(dl), dimension(1:nLat,1:2) :: wavepacket
    integer :: mean, lag
    real(dl) :: width, amplitude, direction

    call gaussian_insertion(wavepacket, mean, width, amplitude, direction, lag)
    fld(:, 1) = fld(:, 1) + wavepacket(:, 1)
    fld(:, 2) = fld(:, 2) + wavepacket(:, 2)
  end subroutine initialize_wavepacket
  
!!!!!!!!!!!!!!
! Wavepacket !
!!!!!!!!!!!!!!


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gaussian_insertion(wavepacket, mean, width, amplitude, direction, lag)
    real(dl), dimension(1:nLat, 1:2), intent(out) :: wavepacket
    real(dl), dimension(1:nLat) :: exponential
    integer :: mean, lag, j
    real(dl) :: a, amplitude, direction, width

    do j = 1, nLat
      a = real(j)
      exponential(j) = exp( - 0.5_dl * (abs(a-real(mean)) / width)**2._dl )
      wavepacket(j, 1) = amplitude * exponential(j)
      wavepacket(j, 2) = direction * (a-real(mean)) / (width)**2._dl * amplitude*real(lag)/real(2) * exponential(j)
    enddo
  end subroutine gaussian_insertion

!!!!!!!!!!!!!!!!
! Fluctuations !
!!!!!!!!!!!!!!!!

  subroutine initialize_vacuum_fluctuations(fld,len,m2,kspec,phi0,klat)
    real(dl), dimension(:,:), intent(inout) :: fld
    real(dl), intent(in) :: len, m2
    integer, intent(in), optional :: kspec, klat
    real(dl), intent(in), optional :: phi0
    
    real(dl), dimension(1:size(fld(:,1)/2+1)) :: spec, w2eff  ! remove w2eff here, it's unneeded
    real(dl), dimension(1:size(fld(:,1))) :: df
    integer :: i,km,kc
    real(dl) :: phiL, norm

    integer :: n, nn; real(dl) :: dk
    dk = twopi / len; n = size(fld(:,1)); nn = n/2+1
    
    km = size(spec); if (present(kspec)) km = kspec
    kc = size(spec); if (present(klat))  kc = klat
    
    phiL = twopi; if (present(phi0)) phiL = phi0

    norm = 1._dl/ phiL / sqrt(4._dl * len) ! second factor of 1/sqrt(2) is normalising the Box-Mueller, first one is from 1/sqrt(2\omega)

    do i=1,nn
       w2eff(i) = m2 + dk**2*(i-1)**2
    enddo
    spec = 0._dl
    spec(2:km) = norm / w2eff(2:km)**0.25
    call generate_1dGRF(df,spec(:kc))
    fld(:,1) = fld(:,1) + df(:)

    spec = spec * w2eff**0.5
    call generate_1dGRF(df,spec(:kc))
    fld(:,2) = fld(:,2) + df(:)
  end subroutine initialize_vacuum_fluctuations

!!!!!!!!!!!!!!!!!!!
! Gaussian Filter !
!!!!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine kspace_gaussian_filter(sm_fft_fld, fft_fld)
    complex(C_DOUBLE_COMPLEX), dimension(1:nyq), intent(out) :: sm_fft_fld
    complex(C_DOUBLE_COMPLEX), dimension(1:nyq), intent(in) :: fft_fld
    real(dl), dimension(1:nLat) :: kernel; integer :: j; real(dl) :: normalisation, a

    normalisation = 1._dl / sqrt(twopi) / sigma
    kernel = 0._dl
    do j = 1, nLat
      if (j < nLat/2+1) then
        a = j-1
      else 
        a = nLat-j+1
      endif
      kernel(j) = exp(-0.5_dl*(a*dx/sigma)**2) * normalisation
    enddo

    tPair%realSpace(:) = kernel(:) / sum(kernel)
    call fftw_execute_dft_r2c(tPair%planf,tPair%realSpace,tPair%specSpace) ! fourier transform
!    print*, "gaussian_window = ["; do n = 1, nyq; print*, tPair%specSpace(n), ',' ; enddo; print*, "] = gaussian_window"
    sm_fft_fld(:) = fft_fld(:) * tPair%specSpace(:)
  end subroutine kspace_gaussian_filter

!!!!!!!!!!!!!!!!!!
! Time Evolution !
!!!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! summons integrator module nTime times; called with dt = dx/alph, ns = 4*n_cross*nLat
  ! writes output files
  subroutine time_evolve(dt, ns, width, amplitude, lag, sim)
    real(dl) :: dt, dtout, amplitude, width; integer :: ns, i, j, outsize, lag, sim

    if (dt > dx) print*, "Warning, violating Courant condition" !i.e. alph > 1
    outsize = ns/nTime ! how much time is needed to compute ncross field intersections or sth like that
    dtout = dt*outsize
    do i = 1, nTime ! this loops over time slices
       do j = 1, outsize ! how many integrations are needed to evolve by one time slice; depends on how many crosses
          call gl10(yvec, dt)
       enddo
       call output_fields(fld, dt, dtout, width, amplitude, lag, sim)
    enddo
  end subroutine time_evolve

!!!!!!!!!!
! Output !
!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Initialise the integrator, setup FFTW, boot MPI, and perform other necessary setup before starting the program
  ! It just calls the initialisation subroutines in fftw and integrator modules
  subroutine setup(nVar)
    integer, intent(in) :: nVar
    call init_integrator(nVar)
    call initialize_transform_1d(tPair,nLat)
  end subroutine setup

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Convert an integer to string.
  character(len=20) function long_real_str(k)
    real(dl), intent(in) :: k
    write (long_real_str, '(f14.8)') k
    long_real_str = adjustl(long_real_str)
  end function long_real_str
  character(len=20) function integer_str(k)
    integer, intent(in) :: k
    write (integer_str, *) k
    integer_str = adjustl(integer_str)
  end function integer_str

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine output_fields(fld, dt, dtout, width, amplitude, lag, sim)
    real(dl), dimension(1:nLat, 1:2) :: fld, sm_fld
    complex(C_DOUBLE_COMPLEX), dimension(1:nyq, 1:2) :: fft_fld, sm_fft_fld
    real(dl) :: dt, dtout, width, amplitude;
    logical :: o; integer :: i, j, m, n, lag, sim !//'_sim'//trim(integer_str(sim))
    integer, parameter :: oFile = 98, ooFile = 99
    inquire(file='/gpfs/dpirvu/collisions/small_fluct_t'//trim(integer_str(nTime))//'_x'//trim(integer_str(nLat))//'_width'//trim(long_real_str(width))//'_amp'//trim(long_real_str(amplitude))//'_lag'//trim(integer_str(lag))//'_sim'//trim(integer_str(sim))//'_fields.dat', opened=o)
    if (.not.o) then
    open(unit=oFile,file='/gpfs/dpirvu/collisions/small_fluct_t'//trim(integer_str(nTime))//'_x'//trim(integer_str(nLat))//'_width'//trim(long_real_str(width))//'_amp'//trim(long_real_str(amplitude))//'_lag'//trim(integer_str(lag))//'_sim'//trim(integer_str(sim))//'_fields.dat')
      write(oFile,*) "# Lattice Parameters n = ", nLat, "dx = ", dx, 'm2eff = ', m2eff, 'len = ', len, 'sigma = ', sigma
      write(oFile,*) "# Time Stepping parameters n = ", nTime, " dt = ", dt, "dt_out = ", dtout
    endif

    call destroy_transform_1d(tPair)
    call initialize_transform_1d(tPair, nLat)

    tPair%realSpace(:) = fld(:, 1) ! store as real part of tPair
    call fftw_execute_dft_r2c(tPair%planf, tPair%realSpace, tPair%specSpace) !direct fft
    fft_fld(:, 1) = tPair%specSpace(:)

    call kspace_gaussian_filter(sm_fft_fld(:, 1), fft_fld(:, 1)) ! smoothen field 
    do i = 1, nyq !for all field modes
       tPair%specSpace(i) = sm_fft_fld(i, 1) / nLat ! add normalisation
    enddo
    call fftw_execute_dft_c2r(tPair%planb, tPair%specSpace, tPair%realSpace) !inverse fft
    sm_fld(:, 1) = tPair%realSpace(:) ! store spectral space smoothened field

    do m = 1, nLat ! for all timeslices
       write(oFile,*) fld(m,1), sm_fld(m,1)
    enddo

  end subroutine output_fields
end program Gross_Pitaevskii_1d
