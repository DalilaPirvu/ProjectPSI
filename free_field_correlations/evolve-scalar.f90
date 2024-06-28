#include "macros.h"
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
!  real(dl) :: phi0 = 1, alph = 8._dl, sigma = 0.01_dl/sqrt(m2eff)
  real(dl) :: phi0 = 1, alph = 8._dl, sigma = 0.02_dl/sqrt(m2eff)
  integer, parameter :: inFile = 70, cpFile = 71 ! parameter type means constant
  integer :: sim, nSims = 5000, n_cross = 2, nTime = 64 ! n_cross = number of bubble intersections; nTime = # instances of propagation

  fld(1:nLat,1:2) => yvec(1:nVar-1) ! store the field in yvec
  time => yvec(nVar) ! last position stores the time?
  call initialize_rand(93286123,12)
  call setup(nVar)

  do sim = 0, nSims-1 ! run nSims simulations for the parameters, each with its output files
      call initialise_fields(fld, nyq, nyq)  ! call fluctuations (fld,kspec,klat)
      call time_evolve(dx/alph, 4*nLat*n_cross, sim)
      print*, "Simulation ", sim+1, " in ", nSims , " done!" 
  enddo
  print*, 'm2eff = ', m2eff, 'len = ', len, 'dt = ', dx/alph, 'dx = ', dx, 'dk = ', dk, 'sigma = ', sigma

contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine initialise_fields(fld, kmax, klat)
    real(dl), dimension(:,:), intent(inout) :: fld
    integer, intent(in) :: kmax, klat

    yvec(nVar) = 0._dl !Add a tcur pointer here
    fld(:,1) = 0._dl ! initialise mean fiend and momentum
    fld(:,2) = 0._dl
    call initialize_vacuum_fluctuations(fld, kmax, klat) ! call fluctuations (fld,kspec,klat)
  end subroutine initialise_fields

!!!!!!!!!!!!!!!!!!
! Time Evolution !
!!!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! summons integrator module nTime times; called with dt = dx/alph, ns = 4*n_cross*nLat
  ! writes output files
  subroutine time_evolve(dt, ns, sim) 
    real(dl) :: dt, dtout; integer :: ns, i, j, outsize, sim
    if (dt > dx) print*, "Warning, violating Courant condition" !i.e. alph > 1
    outsize = ns/nTime ! how much time is needed to compute ncross field intersections or sth like that
    dtout = dt*outsize
    do i = 1, nTime ! this loops over time slices
       do j = 1, outsize ! how many integrations are needed to evolve by one time slice; depends on how many crosses
          call gl10(yvec,dt)
       enddo
       call output_fields(fld, dt, dtout, sim)
    enddo
  end subroutine time_evolve

!!!!!!!!!!!!!!!!
! Fluctuations !
!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine initialize_vacuum_fluctuations(fld, kspec, klat)
  !> Initialise Minkowski Gaussian vacuum approximation for fluctuations
  !> Spectra in this subroutine are truncated for direct comparison of fluctuations generated between lattices of varying size.
    real(dl), dimension(:,:), intent(inout) :: fld
    integer, intent(in), optional :: kspec, klat
    real(dl), dimension(1:nyq) :: spec, w2eff
    real(dl), dimension(1:nLat) :: df
    integer :: i; real(dl) :: norm
    ! The second 1/sqrt(2) is a bug since this factor isn't in my GRF sampler
    norm = 1._dl / phi0 / sqrt(4._dl*len) ! second factor of 1/sqrt(2) is normalising the Box-Mueller, first one is from 1/sqrt(2\omega)
    do i=1, nyq ! for all modes up to n_cutoff = n/2+1; i is the index of the mode, dk the spacing between modes
       w2eff(i) = m2eff + dk**2*(i-1)**2 !frequency at the denominator in random field fluctuation i.e. eq.5 in paper
    enddo
    spec = 0._dl ! k = 1 mode is null
    spec(2:kspec) = norm / w2eff(2:kspec)**0.25 ! 1 < k <= nyq is as in paper
    call generate_1dGRF(df, spec(:klat))
    fld(:,1) = fld(:,1) + df(:) ! generates field fluctuations
    spec = spec * w2eff**0.5
    call generate_1dGRF(df, spec(:klat))
    fld(:,2) = fld(:,2) + df(:) ! generates momentum fluctuations
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
    call fftw_execute_dft_r2c(tPair%planf, tPair%realSpace, tPair%specSpace) ! fourier transform
!    print*, "gaussian_window = ["; do n = 1, nyq; print*, tPair%specSpace(n), ',' ; enddo; print*, "] = gaussian_window"
    sm_fft_fld(:) = fft_fld(:) * tPair%specSpace(:)
  end subroutine kspace_gaussian_filter

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
  ! This is only used to name the output files according to the simulation parameters
  character(len=20) function str(k)
    integer, intent(in) :: k
    write (str, *) k
    str = adjustl(str)
  end function str
  
!!!!!!!!!!
! Output !
!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine output_fields(fld, dt, dtout, sim)
    real(dl), dimension(1:nLat, 1:2) :: fld, sm_fld
    complex(C_DOUBLE_COMPLEX), dimension(1:nyq, 1:2) :: fft_fld, sm_fft_fld
    real(dl) :: dt, dtout; logical :: o; integer :: i, j, m, n, sim
    integer, parameter :: oFile = 98, ooFile = 99

    inquire(file='/gpfs/dpirvu/free_field_correlations/t'//trim(str(nTime))//'_x'//trim(str(nLat))//'_sim'//trim(str(sim))//'_fields.dat', opened=o)
    if (.not.o) then
       open(unit=oFile,file='/gpfs/dpirvu/free_field_correlations/t'//trim(str(nTime))//'_x'//trim(str(nLat))//'_sim'//trim(str(sim))//'_fields.dat')
       write(oFile,*) "# Lattice Parameters n = ", nLat, "dx = ", dx, 'm2eff = ', m2eff, 'len = ', len, 'sigma = ', sigma
       write(oFile,*) "# Time Stepping parameters n = ", nTime, "dt = ", dt, "dt_out = ", dtout
       open(unit=ooFile,file='/gpfs/dpirvu/free_field_correlations/t'//trim(str(nTime))//'_x'//trim(str(nLat))//'_sim'//trim(str(sim))//'_fft_fields.dat')
       write(ooFile,*) "# Lattice Parameters n = ", nLat, "dx = ", dx, 'm2eff = ', m2eff, 'len = ', len, 'sigma = ', sigma
       write(ooFile,*) "# Time Stepping parameters n = ", nTime, "dt = ", dt, "dt_out = ", dtout
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

    call destroy_transform_1d(tPair)
    call initialize_transform_1d(tPair, nLat)

    tPair%realSpace(:) = fld(:, 2) ! store as real part of tPair
    call fftw_execute_dft_r2c(tPair%planf, tPair%realSpace, tPair%specSpace) !direct fft
    fft_fld(:, 2) = tPair%specSpace(:)

    call kspace_gaussian_filter(sm_fft_fld(:, 2), fft_fld(:, 2)) ! smoothen field 
    do j = 1, nyq !for all field modes
       tPair%specSpace(j) = sm_fft_fld(j, 2) / nLat ! add normalisation
    enddo
    call fftw_execute_dft_c2r(tPair%planb, tPair%specSpace, tPair%realSpace) !inverse fft
    sm_fld(:, 2) = tPair%realSpace(:) ! store spectral space smoothened field

    do m = 1, nLat ! for all timeslices
       write(oFile,*) fld(m,:), sm_fld(m,:) ! output field, field momentum, smoothened field
    enddo

!    do n = 1, nyq ! for all timeslices
!       write(ooFile,*) fft_fld(n,:), sm_fft_fld(n,:) ! output fft of field, field momentum, smoothened field
!    enddo
  end subroutine output_fields

end program Gross_Pitaevskii_1d
