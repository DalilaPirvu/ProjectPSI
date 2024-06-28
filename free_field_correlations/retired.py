#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
from matplotlib import gridspec
from labellines import labelLines
import random

#%matplotlib qt
#%matplotlib inline

#locationUbuntu = '/mnt/c/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/improved_Scalar/'
#locationPythonOG = 'C:/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/improved_Scalar/'
correl_plots = './plots/correlations/col2/'
sim_location = './sims/sigma10_t100x90_'
sim_suffix = '_for_sigma10_t100x90'
    
nLat = 90
nTime = 100
nSims = 100
phi0 = 1.
lenLat = nLat
nCols = 4
sigma = 1.
n_cross = 2
nyq = int(nLat/2)+1
hLat = int(nLat/2)
dk = 2.*np.pi/lenLat
m2eff = 1
dx = 1.
alpha = 8.
dt = dx/alpha
outsize = 4*n_cross*nLat/nTime
dtout = dt*outsize

recombination_time = [time for time in range(nTime) if time%(1/(dtout*np.sqrt(m2eff)))==0]
titles = [r'Field $\phi(x)$',r'$\partial_t \phi(x)$',r'Smoothened $\phi(x)$']
fft_titles = [r'$\phi(k)$',r'$\partial_t \phi(k)$',r'Smoothened $\phi(k)$']


############################################################
""" Peak correlation from data """
############################################################

# 1D autocorrelation function
def sm_moment_sq(n):
    """The sum of terms in the discrete fft from k_space giving \sigma_n^2."""
    terms = [(dk*k)**(2*n) * smoothened_pspec(k) for k in range(1, nyq)]
    return sum(terms).real # although I think the th. pspec already contains the normalisation due to FT

# 1D correlation function at reparation R
def sm_correlator(n, R):
    if n % 2 == 0:
        terms = [(dk*k)**(2*n) * smoothened_pspec(k) * np.cos(dk*k*R) for k in range(1, nyq)]
    elif n % 2 == 1:
        terms = [(dk*k)**(2*n) * smoothened_pspec(k) * np.sin(dk*k*R) for k in range(1, nyq)]
    return sum(terms).real

def gamma(n):
    numerator = sm_moment_sq(n)
    denominator = np.sqrt(sm_moment_sq(n-1) * sm_moment_sq(n+1))
    return(numerator / denominator)

gamma = gamma(1)
Rstar = sm_moment_sq(1) / np.sqrt(sm_moment_sq(0)*sm_moment_sq(2))

def A():
    return 5. / 2. / (9. - 5. * gamma**2)

def B_0():
    return 432. / np.sqrt(10. * np.pi) / (9. - 5. * gamma**2)**(5./2.)

def B_1():
    return 4 * B_0() / (9. - 5. * gamma**2)

def G_0(om):
    return om**3 - 3. * gamma**2 * om + B_0() * om**2 * np.exp(- A() * om**2)

def G_1(om):
    return om**4 + 3 * om**2 * (1. - 2. * gamma**2) + B_1() * om**3 * np.exp(- A() * om**2)

def u_bar(thr):
    om = gamma * thr
    return G_1(om) / G_0(om)

def b_nu(thr):
    numerator = thr - gamma * u_bar(thr)
    denominator = 1 - gamma**2
    return numerator / denominator

def b_zeta(thr):
    numerator = u_bar(thr) - gamma * thr
    denominator = 1 - gamma**2
    return numerator / denominator

def b_eta(thr):
    return 2 * gamma * b_nu(thr) * b_zeta(thr)

def th_peak_correlator(thr, R):
#    nu = thr * np.sqrt(m2eff) / np.sqrt(sm_moment_sq(0))
    nu = thr
    return(b_nu(nu)**2 * sm_correlator(0, R) + b_eta(nu) * sm_correlator(1, R) + b_zeta(nu)**2 * sm_correlator(2, R))

def b_10(thr):
    numerator = thr - gamma * u(thr)
    denominator = 1 - gamma**2
    return numerator / denominator / np.sqrt(sm_moment_sq(0))

def b_01(thr):
    numerator = u(thr) - gamma * thr
    denominator = 1 - gamma**2
    return numerator / denominator / np.sqrt(sm_moment_sq(2))

def u(thr):
    numerator = gamma**2 * thr**3 + 3 * thr * (1 - 2 * gamma**2)
    denominator = gamma * thr **2 - 3 * gamma
    return numerator / denominator

def bubble_bias_th_peak_correlator(thr, R):
#    nu = thr * np.sqrt(m2eff) / np.sqrt(sm_moment_sq(0))
    nu = thr
    return(b_10(nu)**2*sm_correlator(0, R) + 2*b_01(nu)*b_10(nu)*sm_correlator(1, R) + b_01(nu)**2*sm_correlator(2, R))

def difference_matrix(a):
    """ Created a matrix of differences of elements of an array. """
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()

def find_peak_positions(col, th, b_size, sim, timeslice):
    """ Finds x coordinate of peaks in smoothened field for mask applied at threshold. """
    peak_positions = []
    smoothened_field_slice = add_mask(col, th, sim)[timeslice]
    peak_positions = scs.find_peaks(smoothened_field_slice, threshold = th, distance = b_size)[0]
    return peak_positions

def peak_position_frequencies(col, th, b_size, timeslice):
    data = [find_peak_positions(col, th, b_size, sim, timeslice) for sim in range(nSims)]
    flat_data = [item for sublist in data for item in sublist] # flattens list of lists
    frequencies = [flat_data.count(R) for R in range(hLat)]
    return(frequencies)

def distances_between_peaks(col, th, b_size, sim, timeslice):
    """ Finds shortest distance (on the circle) between peaks from their x coordinates """
    peak_positions = find_peak_positions(col, th, b_size, sim, timeslice)
    dist_matrix = difference_matrix(peak_positions) # differences should keep into account pbc
    dist_matr = dist_matrix[np.tril_indices(len(dist_matrix))]
    dist_matr = np.asarray([abs(i) for i in dist_matr if i!=0.])
#    dist_matr = [nLat-i if i > hLat else i for i in dist_matr]
    return dist_matr

def peak_separation_frequencies(col, th, b_size, timeslice):
    data = [distances_between_peaks(col, th, b_size, sim, timeslice) for sim in range(nSims)]
    flat_data = [item for sublist in data for item in sublist] # flattens list of lists
    frequencies = [flat_data.count(R) for R in range(hLat)]
    return(frequencies)

def plot_peak_separation_frequencies_at_fixed_time(col, th, b_size, timeslice):
    """Plots histogram of distances between peaks across all simulations."""
    plt.figure(figsize=(7,7))
    separation_frequencies = peak_separation_frequencies(col, th, b_size, timeslice)
    position_frequencies = peak_position_frequencies(col, th, b_size, timeslice)
    separation_norm = np.sum(separation_frequencies)
    position_norm = np.sum(position_frequencies)
    plt.plot(list(range(hLat)), separation_frequencies/separation_norm, 'r-', label = 'Separation')
    plt.plot(list(range(hLat)), position_frequencies/position_norm, 'b-', label = 'Position')
    plt.title(r'Peak frequency')
    plt.ylabel(r'Frequency')
    plt.grid()
    plt.savefig(correl_plots + 'peak_separation_frequency_across_sims_at_t_'+str(timeslice)+ '_col' + str(col) + plot_sim_suffix + '.png')
    plt.show()
    return

############################################################################################

def distance_matrix():
    """Finds all distances between fields at any point on the lattice"""
    dist = np.asarray([np.asarray([None for x1 in range(nLat)]) for x2 in range(nLat)])
    for x1 in range(nLat):
        for x2 in range(nLat):
            dist[x1][x2] = np.abs(x1-x2)
    dist_matr = dist[np.triu_indices(len(dist))]
    return np.asarray([min(i,nLat-i) for i in dist_matr])

def rspec(col, sim, timeslice):
    """Computes \phi(x_1)\phi(x_2) between all points on the lattice at a particular time and simulation."""
    field = all_real[sim][col][timeslice]
    amp = np.asarray([np.asarray([None for x1 in range(nLat)]) for x2 in range(nLat)])
    for x1 in range(nLat):
        for x2 in range(nLat):
            amp[x1][x2] = field[x1]*field[x2]
    return amp[np.triu_indices(len(amp))]

def rspec_average_for_fixed_R(col, sim, timeslice, R):
    """For a particular distance, it computes the average of all field amplitude products on the lattice."""
    amp = rspec(col, sim, timeslice)
    dist = distance_matrix()
    res_list = list(filter(lambda x: dist[x] == R, range(len(dist))))
    return np.mean([amp[i] for i in res_list])

def rspec_overall_average(col, R):
    return np.mean([rspec_average_for_fixed_R(col, sim, timeslice, R) for sim in range(nSims) for timeslice in range(nTime)])

def rspec_sim_average(col, timeslice, R):
    return np.mean([rspec_average_for_fixed_R(col, sim, timeslice, R) for sim in range(nSims)])


def time_distance_matrix():
    """Finds all distances between fields at any point on the lattice"""
    dist = np.asarray([np.asarray([None for t1 in range(nTime)]) for t2 in range(nTime)])
    for t1 in range(nTime):
        for t2 in range(nTime):
            dist[t1][t2] = t1-t2
    dist_matr = dist[np.tril_indices(len(dist))]
    return dist_matr

def data_rspace_equal_site(col, sim, spaceslice):
    time_field = [all_real[sim][col][i][spaceslice] for i in range(nTime)]
    #corresponds to: a simulation, col = field data, i = timeslice [0,nTime-1], at fixed spaceslice = lattice site
    amp = np.asarray([np.asarray([None for t1 in range(nTime)]) for t2 in range(nTime)])
    for t1 in range(nTime):
        for t2 in range(nTime):
            amp[t1][t2] = time_field[t1]*time_field[t2]
    return amp[np.tril_indices(len(amp))]

def average_for_fixed_deltat(col, sim, spaceslice, deltat):
    """For a particular time interval, it computes the average of all field amplitude products on the lattice."""
    amp = data_rspace_equal_site(col, sim, spaceslice)
    dist = time_distance_matrix()
    res_list = list(filter(lambda t: dist[t] == deltat, range(len(dist))))
    return np.mean([amp[i] for i in res_list])

def average_for_fixed_deltat_overall(col, deltat):
    return np.mean([average_for_fixed_deltat(col, sim, spaceslice, deltat) for sim in range(nSims) for spaceslice in range(1,nLat,50)])

def average_for_fixed_deltat_and_simulations(col, spaceslice, deltat):
    return np.mean([average_for_fixed_deltat(col, sim, spaceslice, deltat) for sim in range(nSims)])


def timelike_two_point_function(col):
    """Plots autocorrelation function at all time intervals, averaged over lattice and simulations."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    if col == 0:
        ax0.plot(list(range(nTime)),[timelike_fft_pspec(deltat) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>_{data}$ from $P(k)_{th}$')
    elif col == 2:
        ax0.plot(list(range(nTime)),[timelike_fft_smoothened_pspec(deltat, sigma) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>$ from smoothened $ P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    rspec_data = [average_for_fixed_deltat_overall(col, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), rspec_data, 'ko' , label = '$<\phi(x,t_1)\phi(x,t_2)>_{data}$')

    pspec_data = [time_fft_data_pspec_sim_averaged(col, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), pspec_data, 'go', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}$')

    [ax0.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    [ax1.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    ax1.plot(list(range(nTime)), [average_for_fixed_deltat_overall(col, deltat)-timelike_fft_pspec(deltat) for deltat in range(nTime)], color='crimson', ls='--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{data}-<\phi(x,t_1)\phi(x,t_2)>_{th}$')
    ax1.plot(list(range(nTime)), [rspec_data[i]-pspec_data[i] for i in range(nTime)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(nTime)), np.zeros(nTime), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta t = |t_1-t_2|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point timelike correlation function', fontsize=14)
    plt.savefig(correl_plots+'timelike_two_point_function'+sim_suffix+'.png')
    return

def timelike_two_point_function_at_lattice_position(col, spaceslice):
    """Plots autocorrelation function at all time intervals, averaged over lattice and simulations."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    if col == 2:
        ax0.plot(list(range(nTime)),[timelike_fft_smoothened_pspec(deltat, sigma) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>$ from smoothened $P(k)_{th}$')
    elif col == 0:
        ax0.plot(list(range(nTime)),[timelike_fft_pspec(deltat) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>_{data}$ from $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    rspec_data = [average_for_fixed_deltat_and_simulations(col, spaceslice, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), rspec_data, 'ko' , label = '$<\phi(x,t_1)\phi(x,t_2)>_{data}$ at x='+str(spaceslice))

    pspec_data = [time_fft_data_pspec_sim_averaged(col, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), pspec_data, 'go', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}$')

    [ax0.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    [ax1.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    ax1.plot(list(range(nTime)), [average_for_fixed_deltat_and_simulations(col, spaceslice, deltat)-timelike_fft_pspec(deltat) for deltat in range(nTime)], color='crimson', ls='--',  label = r'$<\phi(x,t_1)\phi(x,t_2)>_{data}-<\phi(x,t_1)\phi(x,t_2)>_{th}$')
    ax1.plot(list(range(nTime)), [rspec_data[i]-pspec_data[i] for i in range(nTime)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(nTime)), np.zeros(nTime), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta t = |t_1-t_2|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point timelike correlation function at x='+str(spaceslice), fontsize=14)
    plt.savefig(correl_plots+'timelike_two_point_function_at_lattice_position'+str(spaceslice)+sim_suffix+'.png')
    return

def timelike_two_point_function_at_lattice_position_at_sim(col, sim, spaceslice):
    """Plots autocorrelation function at all time intervals, averaged over lattice and simulations."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    if col == 2:
        ax0.plot(list(range(nTime)),[timelike_fft_smoothened_pspec(deltat, sigma) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>$ from smoothened $P(k)_{th}$')
    elif col == 0:
        ax0.plot(list(range(nTime)),[timelike_fft_pspec(deltat) for deltat in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>_{data}$ from $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    rspec_data = [average_for_fixed_deltat(col, sim, spaceslice, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), rspec_data, 'ko' , label = '$<\phi(x,t_1)\phi(x,t_2)>_{data}$ at x='+str(spaceslice)+', sim ='+str(sim))

    pspec_data = [time_fft_data_pspec(col, sim, deltat) for deltat in range(nTime)]
    ax0.plot(list(range(nTime)), pspec_data, 'go', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}$ at sim='+str(sim))

    [ax0.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    [ax1.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    ax1.plot(list(range(nTime)), [average_for_fixed_deltat(col, sim, spaceslice, deltat)-timelike_fft_pspec(deltat) for deltat in range(nTime)], color='crimson', ls='--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{data}-<\phi(x,t_1)\phi(x,t_2)>_{th}$')
    ax1.plot(list(range(nTime)), [rspec_data[i]-pspec_data[i] for i in range(nTime)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(nTime)), np.zeros(nTime), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta t = |t_1-t_2|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point timelike correlation function at x='+str(spaceslice)+', sim='+str(sim), fontsize=14)
    plt.savefig(correl_plots+'timelike_two_point_function_at_lattice_position'+str(spaceslice)+'_at_sim'+str(sim)+sim_suffix+'.png')
    return

def spacelike_two_point_function(col):
    """Plots autocorrelation function (from the power spectrum) at all distances, averaged in time."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    rspec_data = [rspec_overall_average(col, R) for R in range(hLat)]
    ax0.plot(list(range(hLat)), rspec_data, 'ko' , label = '$<\phi(x,t)\phi(y,t)>_{data}$ from rspec data')

    pspec_data = [fft_data_pspec_averaged(col, R) for R in range(nLat)]
    pspec_data = [0.5*(pspec_data[i]+pspec_data[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    ax0.plot(list(range(hLat)), pspec_data, 'go', label = r'$<\phi(x,t)\phi(y,t)>_{P(k)}$')

    if col == 0:
        th = [spacelike_fft_pspec(R) for R in range(nLat)]
        th = [0.5*(th[i] + th[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
        ax0.plot(list(range(hLat)), th ,'r--', label = r'FFT $P(k)_{th}$')
    elif col == 2:
        thh = [spacelike_fft_smoothened_pspec(R, sigma) for R in range(nLat)]
        thh = [0.5*(thh[i] + thh[nLat-1:hLat-1:-1][i]) for i in range(hLat)]      
        ax0.plot(list(range(hLat)), thh ,'r--', label = r'FFT smoothened $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    ax1.plot(list(range(hLat)), [rspec_data[i]-pspec_data[i] for i in range(hLat)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(hLat)), np.zeros(hLat), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta R = |x-y|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point spacelike correlation function', fontsize=14)
    plt.savefig(correl_plots+'spacelike_two_point_function'+sim_suffix+'.png')
    return

def spacelike_two_point_function_at_timeslice(col, timeslice):
    """Plots autocorrelation function (from the power spectrum) at all distances, at fixed time."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    rspec_data = [rspec_sim_average(col, timeslice, R) for R in range(hLat)]
    ax0.plot(list(range(hLat)), rspec_data, 'ko' , label = '$<\phi(x,t)\phi(y,t)>_{data}$ from rspec data at t='+str(timeslice))

    pspec_data = [fft_data_pspec_sim_averaged(col, timeslice, R) for R in range(nLat)]
    pspec_data = [0.5*(pspec_data[i]+pspec_data[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    ax0.plot(list(range(hLat)), pspec_data, 'go', label = r'$<\phi(x,t)\phi(y,t)>_{P(k)}$ at t='+str(timeslice))

    if col == 0:
        th = [spacelike_fft_pspec(R) for R in range(nLat)]
        th = [0.5*(th[i] + th[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
        ax0.plot(list(range(hLat)), th ,'r--', label = r'FFT $P(k)_{th}$')
    elif col == 2:
        thh = [spacelike_fft_smoothened_pspec(R, sigma) for R in range(nLat)]
        thh = [0.5*(thh[i] + thh[nLat-1:hLat-1:-1][i]) for i in range(hLat)]      
        ax0.plot(list(range(hLat)), thh ,'r--', label = r'FFT smoothened $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    ax1.plot(list(range(hLat)), [rspec_data[i]-pspec_data[i] for i in range(hLat)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(hLat)), np.zeros(hLat), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta R = |x-y|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point spacelike correlation function at t='+str(timeslice), fontsize=14)
    plt.savefig(correl_plots+'spacelike_two_point_function_at_timeslice_'+str(timeslice)+sim_suffix+'.png')
    return

def spacelike_two_point_function_at_timeslice_at_sim(col, sim, timeslice):
    """Plots autocorrelation function at all distances, at fixed time."""
    fig = plt.figure(figsize=(12,8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    rspec_data = [rspec_average_for_fixed_R(col, sim, timeslice, R) for R in range(hLat)]
    ax0.plot(list(range(hLat)), rspec_data, 'ko' , label = r'$<\phi(x,t)\phi(y,t)>_{data}$ at t='+str(timeslice)+', sim='+str(sim))

    pspec_data = [fft_data_pspec(col, sim, timeslice, R) for R in range(nLat)]
    pspec_data = [0.5*(pspec_data[i]+pspec_data[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    ax0.plot(list(range(hLat)), pspec_data, 'go', label = r'$<\phi(x,t)\phi(y,t)>_{P(k)}$ at t='+str(timeslice)+', sim='+str(sim))

    if col == 0:
        th = [spacelike_fft_pspec(R) for R in range(nLat)]
        th = [0.5*(th[i] + th[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
        ax0.plot(list(range(hLat)), th ,'r--', label = r'FFT $P(k)_{th}$')
    elif col == 2:
        thh = [spacelike_fft_smoothened_pspec(R, sigma) for R in range(nLat)]
        thh = [0.5*(thh[i] + thh[nLat-1:hLat-1:-1][i]) for i in range(hLat)]      
        ax0.plot(list(range(hLat)), thh ,'r--', label = r'FFT smoothened $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    ax0.set_ylabel(r'$<\phi\phi>$')

    ax1.plot(list(range(hLat)), [rspec_data[i]-pspec_data[i] for i in range(hLat)], 'c--',label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(hLat)), np.zeros(hLat), color='grey', ls='--')

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax1.set_xlabel(r'$\Delta R = |x-y|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point spacelike correlation function at t='+str(timeslice)+', sim='+str(sim), fontsize=14)
    plt.savefig(correl_plots+'spacelike_two_point_function_at_timeslice_'+str(timeslice)+'_at_sim_'+str(sim)+sim_suffix+'.png')
    return

############################################################
""" Field-field correlation from spectral data"""
############################################################

def data_pspec(col, sim, timeslice, k):
    """ This is the P(k)=P(-k) from data for k = 0,1,...,nLat/2+1=nyq"""
    slices = all_spec[sim][col][timeslice] 
    return np.abs(slices[k])**2

def data_pspec_averaged(col, k):
    return np.mean([data_pspec(col, sim, timeslice, k) for sim in range(nSims) for timeslice in range(nTime)])

def fft_data_pspec(col, sim, timeslice, R):
    """The sum of terms in the discrete fft from k_space oower spectrum to x_space autocorrelation function at separation R."""
    terms = [data_pspec(col, sim, timeslice, k)*np.exp(-1j*dk*k*R) for k in range(nyq)]
    return sum(terms)

def fft_data_pspec_sim_averaged(col, timeslice, R):
    """Returns average value across all simulations at fixed time of the autocorrelation function at separation R."""
    return np.mean([fft_data_pspec(col, sim, timeslice, R) for sim in range(nSims)])

def fft_data_pspec_averaged(col, R):
    """Returns average value across all simulations and all time of the autocorrelation function at separation R."""
    return np.mean([fft_data_pspec(col, sim, timeslice, R) for sim in range(nSims) for timeslice in range(nTime)])

def data_rspec(col, sim, time, k):
    """ This is the P(k)=P(-k) from data for k = 0,1,...,nLat/2+1=nyq"""
    time_field = all_spec[sim][col][time]
    return np.abs(time_field[k])**2

def data_rspec_averaged(col, k):
    return np.mean([data_rspec(col, sim, time, k) for sim in range(nSims) for time in range(nTime)])

def time_fft_data_pspec(col, deltat):
    """The sum of terms in the discrete fft from k_space power spectrum to time-time autocorrelation function at separation \Delta t."""
    sim = random.randint(0,nSims-1)
    return sum([data_rspec(col, sim, deltat, k)*np.exp(-1j*omega_k(k)*deltat*dtout) for k in range(nyq)])

""""""""""""""""""""
" Helper Functions "
""""""""""""""""""""

def complex_converter(txt):
    ii, jj = np.safe_eval(txt)
    return np.complex(ii, jj)

def extract_data(filename, col, sim):
    infile = open(filename,'r')
    lines = infile.readlines()
    field_values = [float(line.split()[col]) for line in lines[2:]]
    infile.close()
    return field_values

def extract_fft_data(filename, col, sim):
    infile = open(filename,'r')
    lines = infile.readlines()
    str_values = [line.split()[col] for line in lines[2:]]
    complex_values = [complex_converter(str_values[i]) for i in range(len(str_values))]
    infile.close()
    return complex_values

def data_from_files():
    sims_tableau = []
    for sim in range(nSims):
        all_values,tableau = [],[]
        for col in range(nCols):
            all_values.append(extract_data(sim_location + 'fields' + str(sim) + '.dat', col, sim))
            tableau.append(np.reshape(all_values[col],(nTime,nLat)))
        sims_tableau.append(tableau)
    print('Data format: ', np.shape(sims_tableau),' and corresponds to: [simulation,variable, time slice, amplitude]')
    print('Variables in order: field, time derivative, smoothened field')
    return(sims_tableau)

def data_from_fft_files():
    sims_tableau = []
    for sim in range(nSims):
        all_values,tableau = [],[]
        for col in range(nCols):
            all_values.append(extract_fft_data(sim_location + 'fft_fields' + str(sim) + '.dat', col, sim))
            tableau.append(np.reshape(all_values[col],(nTime,nyq)))
        sims_tableau.append(tableau)
    print('Data format: ', np.shape(sims_tableau),' and corresponds to: [simulation, variable, frequency, amplitude]')
    print('Variables in order: fft field, fft field momentum, fft smoothened field')
    return(sims_tableau)
    
all_real = data_from_files()
all_spec = data_from_fft_files()

######################################################################################## 
""" Analytic Quantities """
######################################################################################## 

def omega_k(k):
    return ((dk*k)**2 + m2eff)**(0.5)

def spectral_scalar_field(k):
    #note 1/len normalisation is from the fourier transform below
    return 1./phi0/np.sqrt(2*omega_k(k)*lenLat)

def window_gaussian(k, sigma):
    return np.exp(-(dk*k*sigma)**2/2.)

def pspec(k):
    return (spectral_scalar_field(k))**2

def smoothened_pspec(k,sigma):
    return (window_gaussian(k, sigma)*spectral_scalar_field(k))**2

def th_pspec():
    plt.figure(figsize=(4,4))
    plt.plot(list(range(nyq)),[smoothened_pspec(k,sigma) for k in range(nyq)],label=r'Smoothened $P(k)_{th}$')
    plt.plot(list(range(nyq)),[pspec(k) for k in range(nyq)],'r', label=r'$P(k)_{th}$')
    plt.xlabel(r'k_n')  
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots + 'spacelike_th_pspec'+sim_suffix+'.png')
    return

######################################################################################## 

def spacelike_fft_smoothened_pspec(R, sigma):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [smoothened_pspec(k,sigma)*np.exp(-1j*dk*k*R) for k in range(nyq)]
    return sum(terms)

def spacelike_fft_pspec(R):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [pspec(k)*np.exp(-1j*dk*k*R) for k in range(nyq)]
    return sum(terms)

def timelike_fft_smoothened_pspec(t, sigma):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [smoothened_pspec(k,sigma)*np.exp(-1j*omega_k(k)*dtout*t) for k in range(nyq)]
    return sum(terms)

def timelike_fft_pspec(t):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [pspec(k)*np.exp(-1j*omega_k(k)*dtout*t) for k in range(nyq)]
    return sum(terms)