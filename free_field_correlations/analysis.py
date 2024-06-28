############################################################
# Activate when using cluster ##############################  
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
from matplotlib import gridspec
from labellines import labelLines
import random
plt.rcParams.update({'font.size': 16})

#%matplotlib qt
#%matplotlib inline

############################################################
# Always check parameters below  ###########################  
nLat = 200
nTime = 200
nSims = 70

m2eff = 1.
phi0 = 1.
dx = 1.
lenLat = nLat
alpha = 8.
nCols = 4
n_cross = 2
sigma = 3.

############################################################
# Check parameters below ###################################
nyq = int(nLat/2)+1
hLat = int(nLat/2)
dk = 2.*np.pi/lenLat
dt = dx/alpha
outsize = 4*n_cross*nLat/nTime
dtout = dt*outsize

#recombination_time = [time for time in range(nTime) if time%(1/(dtout*np.sqrt(m2eff)))==0]

titles = [r'Field $\phi(x)$',r'$\partial_t \phi(x)$',r'Smoothened $\phi(x)$']
fft_titles = [r'$\phi(k)$',r'$\partial_t \phi(k)$',r'Smoothened $\phi(k)$']

locationUbuntu = '/mnt/c/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/free_field_correlations/'
locationPythonOG = 'C:/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/free_field_correlations/'

correl_plots = './plots/correlations/'
field_plots = './plots/field/'
plot_sim_suffix = '_for_t'+str(nTime)+'_x'+str(nLat)+'_fields'
plot_fft_sim_suffix = '_for_t'+str(nTime)+'_x'+str(nLat)+'_fft_fields'

############################################################
""" Peak-peak correlation """
############################################################

def peak_peak_correlation(col, thresh, b_size):
    """Plots histogram of distances between peaks across all simulations."""
    plt.figure(figsize=(12,12))
    data = [find_bubble_peaks(col, thresh, b_size, sim) for sim in range(nSims)]
    flat_data = [item for sublist in data for item in sublist] # flattens list of lists
    frequencies = [flat_data.count(R) for R in range(hLat)]
    plt.plot(list(range(hLat)), frequencies, 'r-')
    plt.title('Peak-peak correlation')
    plt.xlabel(r'$\Delta R$')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(correl_plots + 'peak_peak_correlation'+'_col'+str(col)+plot_sim_suffix+'.png')
    plt.show()
    return

########################################################################################

def find_bubble_peaks(col, thresh, b_size, sim):
    peak_positions = []
    timeslice = random.randint(0, nTime-1)
    masked_field = add_mask(col, thresh)
    smoothened_field_slice = masked_field[sim][timeslice]
    peak_positions = scs.find_peaks(smoothened_field_slice, threshold = 0., distance = b_size)[0]
    return peak_positions

def peak_statistics(col, thresh, b_size, sim):
    peak_positions = find_bubble_peaks(col, thresh, b_size, sim)
    dist_matrix = difference_matrix(peak_positions) # differences should keep into account pbc
    dist_matr = dist_matrix[np.tril_indices(len(dist_matrix))]
    dist_matr = np.asarray([abs(i) for i in dist_matr if i!=0])
    dist = [i if i < hLat else nLat-i for i in dist_matr]
    return dist

def difference_matrix(a):
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()
    
def th_moment(n):
    """The sum of terms in the discrete fft from k_space giving \sigma_n^2."""
    terms = [(spectral_scalar_field(k)*window_gaussian(k, sigma))**2*(dk*k)**(2*n) for k in range(nyq)]
    return np.sqrt(2/lenLat*sum(terms))

def gamma(n):
    return th_moment(n)**2/th_moment(n-1)/th_moment(n-2)

def R_n(n):
    """ R0 = typical separation between zero-crossings of the density field,
    R1 = mean distance between extrema. In general R_n are characteristic length scales."""
    return np.sqrt(3)*th_moment(n)/th_moment(n+1)

def th_peak_density(thresh):
    norm = (lenLat)**(-2)*(gamma(1)/R_n(1))**3
    return norm*(thresh**2-1)*np.exp(-thresh**2/2)
   
############################################################
""" Plotting correlation functions """
############################################################

def spacelike_fft_th_pspec():
    plt.figure(figsize=(12,12))
    th = [spacelike_fft_pspec(R) for R in range(nLat)]
    th_sm = [spacelike_fft_smoothened_pspec(R, sigma) for R in range(nLat)]
    th = [0.5*(th[i] + th[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    th_sm = [0.5*(th_sm[i] + th_sm[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    plt.plot(list(range(hLat)), th, 'r', label=r'FFT $P(k)_{th}$')
    plt.plot(list(range(hLat)), th_sm, label=r'FFT smoothened $P(k)_{th}$')
    plt.xlabel(r'$k_n$')  
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots +'spacelike_fft_th_corr'+plot_sim_suffix+'.png')
    plt.show()
    return

def timelike_fft_th_pspec():
    plt.figure(figsize=(12,12))
    plt.plot(list(range(nTime)),[timelike_fft_pspec(t) for t in range(nTime)],'b', label=r'Smoothened $<\phi(x,t_1)\phi(x,t_2)>_{th}$')
    plt.plot(list(range(nTime)),[timelike_fft_smoothened_pspec(t, sigma) for t in range(nTime)],'r', label=r'$<\phi(x,t_1)\phi(x,t_2)>_{th}$')
    plt.xlabel(r'$\Delta t=t_1-t_2$')  
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots+'timelike_pspec_th'+plot_sim_suffix+'.png')
    plt.show()
    return

def space_pspec_th_and_data(col):
    plt.figure(figsize=(12, 12))
    eff_pspec = [data_pspec_sim_averaged(col, k) for k in range(nyq)]
    plt.plot(list(range(nyq)), eff_pspec, 'bo', label=r'$P(k)_{data}$')
    if col == 0:
        th_pspec = [pspec(k) for k in range(nyq)]    
        plt.plot(list(range(nyq)), th_pspec, 'r--',label=r'$P(k)_{th}$')
    elif col == 2:
        th_pspec = [smoothened_pspec(k, sigma) for k in range(nyq)] 
        plt.plot(list(range(nyq)), th_pspec, 'r--',label=r'Smoothened $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots + 'space_pspec_th_and_data'+'_col'+str(col)+plot_sim_suffix+'.png')
    plt.show()
    return

def time_pspec_th_and_data(col):
    plt.figure(figsize=(12, 12))
    eff_pspec = [data_pspec_sim_averaged(col, k) for k in range(nyq)]
    plt.plot(list(range(nyq)), eff_pspec, 'bo', label=r'$P(k)_{data}$')
    if col == 0:
        th_pspec = [pspec(k) for k in range(nyq)]    
        plt.plot(list(range(nyq)), th_pspec, 'r--',label=r'$P(k)_{th}$')
    elif col == 2:
        th_pspec = [smoothened_pspec(k, sigma) for k in range(nyq)] 
        plt.plot(list(range(nyq)), th_pspec, 'r--',label=r'Smoothened $P(k)_{th}$')
    else:
        print('col = 0 gives the field, col = 2 gives the Gaussian smoothed field')
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots + 'time_pspec_th_and_data'+'_col'+str(col)+plot_sim_suffix+'.png')
    plt.show()
    return

def spacelike_two_point_function(col, x_ref):
    """Plots autocorrelation function at all distances, averaged in time."""
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    if col == 0:
        th = [spacelike_fft_pspec(R) for R in range(nLat)]
        th = [0.5*(th[i] + th[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
        ax0.plot(list(range(hLat)), th ,'r--', label = r'$<\phi(x,t)\phi(y,t)>$ from $P(k)_{th}$')
    elif col == 2:
        thh = [spacelike_fft_smoothened_pspec(R, sigma) for R in range(nLat)]
        thh = [0.5*(thh[i] + thh[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
        ax0.plot(list(range(hLat)), thh ,'r--', label = r'$<\phi(x,t)\phi(y,t)>$ from smoothened $P(k)_{th}$')

    rspec_data = [sim_average_of_space_difference_amp(col, x_ref, R) for R in range(nLat)] # not sure it works well for x_ref > 0
    rspec_data = [0.5*(rspec_data[i]+rspec_data[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    ax0.plot(list(range(hLat)), rspec_data, 'ko' , label = '$<\phi(x,t)\phi(y,t)>_{data}$')

    pspec_data = [pspec_fft_space_sim_averaged(col, R) for R in range(nLat)]
    pspec_data = [0.5*(pspec_data[i]+pspec_data[nLat-1:hLat-1:-1][i]) for i in range(hLat)]
    ax0.plot(list(range(hLat)), pspec_data, 'go', label = r'$<\phi(x,t)\phi(y,t)>_{P(k)}$')

    ax1.plot(list(range(hLat)), [rspec_data[i]-pspec_data[i] for i in range(hLat)], 'c--', label = r'$<\phi(x,t)\phi(y,t)>_{P(k)}-<\phi(x,t)\phi(y,t)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(hLat)), np.zeros(hLat), color='grey', ls='--') 

    print(np.mean(rspec_data)-np.mean(pspec_data))
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax0.set_ylabel(r'$<\phi\phi>$')
    ax1.set_xlabel(r'$\Delta R = |x-y|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point spacelike correlation function', fontsize=24)
    plt.savefig(correl_plots+'spacelike_two_point_function'+'_col'+str(col)+plot_sim_suffix+'.png')
    plt.show()
    return

def timelike_two_point_function(col, t_ref):
    """Plots autocorrelation function at all time intervals, averaged over lattice and simulations."""
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    if col == 0:
        ax0.plot(list(range(nTime-t_ref)),[timelike_fft_pspec(deltat) for deltat in range(nTime-t_ref)],'r--', label=r'$<\phi(x,t_1)\phi(x,t_2)>$ from $P(k)_{th}$')
    elif col == 2:
        ax0.plot(list(range(nTime-t_ref)),[timelike_fft_smoothened_pspec(deltat, sigma) for deltat in range(nTime-t_ref)],'r--', label=r'$<\phi(x,t_1)\phi(x,t_2)>$ from smoothened $ P(k)_{th}$')

    rspec_data = [sim_average_of_time_difference_amp(col, t_ref, deltat) for deltat in range(nTime-t_ref)]
    ax0.plot(list(range(nTime-t_ref)), rspec_data, 'ko' , label = '$<\phi(x,t_1)\phi(x,t_2)>_{data}$')

    pspec_data = [pspec_fft_time_sim_averaged(col, deltat) for deltat in range(nTime-t_ref)]
    ax0.plot(list(range(nTime-t_ref)), pspec_data, 'go', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}$')

    ax1.plot(list(range(nTime-t_ref)), [rspec_data[i]-pspec_data[i] for i in range(nTime-t_ref)], 'c--', label = r'$<\phi(x,t_1)\phi(x,t_2)>_{P(k)}-<\phi(x,t_1)\phi(x,t_2)>_{data}$')
    ax1.set_ylabel(r'$\Delta <\phi\phi>$')
    ax1.plot(list(range(nTime-t_ref)), np.zeros(nTime-t_ref), color='grey', ls='--')
    print(np.mean(rspec_data)-np.mean(pspec_data))
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax0.set_ylabel(r'$<\phi\phi>$')
    ax1.set_xlabel(r'$\Delta t = |t_1-t_2|$')
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    plt.subplots_adjust(hspace=.0)
    fig.suptitle(r'Two-point timelike correlation function, $t_{ref} = $'+str(t_ref), fontsize=24)
    plt.savefig(correl_plots+'timelike_two_point_function_at_tref='+str(t_ref)+'_col'+str(col)+plot_sim_suffix+'.png')
    plt.show()
    return

############################################################
""" Field-field correlation from spectral data"""
############################################################

def data_pspec(col, sim, k):
    """ This is the P(k)=P(-k) from data for k = 0,1,...,nLat/2+1=nyq"""
    timeslice = random.randint(0, nTime-1)
    spec_field = all_spectral_space[sim][col][timeslice]
    return spec_field[k]*np.conj(spec_field[k])

def data_pspec_sim_averaged(col, k):
    return np.mean([data_pspec(col, sim, k) for sim in range(nSims)])

def pspec_fft_space(col, sim, R):
    """The sum of terms in the discrete fft from k_space oower spectrum to x_space autocorrelation function at separation R."""
    terms = [data_pspec(col, sim, k)*2*np.cos(dk*k*R) for k in range(nyq)] 
    return np.sum(terms)

def pspec_fft_time(col, sim, deltat):
    """The sum of terms in the discrete fft from k_space power spectrum to time-time autocorrelation function at separation \Delta t."""
    terms = [data_pspec(col, sim, k)*2*np.cos(omega_k(k)*deltat*dtout) for k in range(nyq)]
    return np.sum(terms)
#(phi0**2*2*lenLat*data_pspec(col, sim, k))**(-0.5)

def pspec_fft_space_sim_averaged(col, R):
    """Returns average value across all simulations at fixed time of the autocorrelation function at separation R."""
    return np.mean([pspec_fft_space(col, sim, R) for sim in range(nSims)])

def pspec_fft_time_sim_averaged(col, deltat):
    """Returns average value across all simulations at fixed time of the autocorrelation function at separation R."""
    return np.mean([pspec_fft_time(col, sim, deltat) for sim in range(nSims)])

############################################################
""" Spacelike two-point correlation from real space data"""
############################################################

def space_difference_amplitude(col, sim, timeslice, x_ref, R):
    field = all_real_space[sim][col][timeslice]
    if R >= hLat:
        R = nLat - R
    return field[x_ref] * field[x_ref + R]

def sim_average_of_space_difference_amp(col, x_ref, R):
    timeslice = random.randint(0, nTime-1)
    return np.mean([space_difference_amplitude(col, sim, timeslice, x_ref, R) for sim in range(nSims)])

def time_difference_amplitude(col, sim, spaceslice, t_ref, deltat):
    time_field = [all_real_space[sim][col][i][spaceslice] for i in range(nTime)]
    return time_field[t_ref] * time_field[t_ref + deltat]

def sim_average_of_time_difference_amp(col, t_ref, deltat):
    spaceslice = random.randint(0, nLat-1)
    return np.mean([time_difference_amplitude(col, sim, spaceslice, t_ref, deltat) for sim in range(nSims)])

######################################################################################## 
""" Analytic Quantities """
######################################################################################## 

def omega_k(k):
    return ((dk*(k-1))**2 + m2eff)**(0.5)

def spectral_scalar_field(k):
    #note 1/len normalisation is from the fourier transform below
    return 1./phi0/np.sqrt(2*omega_k(k)*lenLat)

def window_gaussian(k, sigma):
    return np.exp(-0.5*(dk*k*sigma)**2)

def pspec(k):
    return (spectral_scalar_field(k))**2

def smoothened_pspec(k, sigma):
    return (window_gaussian(k, sigma)*spectral_scalar_field(k))**2

def th_pspec():
    plt.figure(figsize=(12,12))
    plt.plot(list(range(nyq)),[smoothened_pspec(k, sigma) for k in range(nyq)], 'bo', label=r'Smoothened $P(k)_{th}$')
    plt.plot(list(range(nyq)),[pspec(k) for k in range(nyq)], 'ro', label=r'$P(k)_{th}$')
    plt.xlabel(r'$k_n$')  
    plt.grid()
    plt.legend()
    plt.savefig(correl_plots+'spacelike_th_pspec'+plot_sim_suffix+'.png')
    return

######################################################################################## 

def spacelike_fft_smoothened_pspec(R, sigma):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [smoothened_pspec(k,sigma)*2*np.cos(dk*k*R) for k in range(nyq)]
    return sum(terms)

def spacelike_fft_pspec(R):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [pspec(k)*2*np.cos(dk*k*R) for k in range(nyq)]
    return sum(terms)

def timelike_fft_smoothened_pspec(t, sigma):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [smoothened_pspec(k,sigma)*2*np.cos(omega_k(k)*dtout*t) for k in range(nyq)]
    return sum(terms)

def timelike_fft_pspec(t):
    """The sum of terms in the discrete fft of theoretical k_space power spectrum."""
    terms = [pspec(k)*2*np.cos(omega_k(k)*dtout*t) for k in range(nyq)]
    return sum(terms)

############################################################
""" VISUALISING SIMULATION DATA """
############################################################

def plot_real_space_data(col):
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4),sharex='col')
    for sim in range(nSims):
        tableau = all_real_space[sim][col]
        im = ax[sim].imshow(tableau, aspect='auto', interpolation='none', origin='lower')
        plt.colorbar(im, ax = ax[sim])
        ax[sim].set(xlabel=r'$dx$', ylabel=r'$dt$')
        ax[sim].set_title('Simulation ' + str(sim+1) + ': ' + titles[col])
    plt.savefig(field_plots+'_col'+str(col)+plot_sim_suffix+'.png')
    return

def plot_specral_space_data(col):
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4),sharex='col')
    for sim in range(nSims):
        tableau = all_spectral_space[sim][col]
        pspec = [[abs(i*np.conj(i)) for i in tableau[j]] for j in range(nLat)]
        im = ax[sim].imshow(pspec, aspect='auto', interpolation='none', origin='lower')
        plt.colorbar(im, ax = ax[sim])
        ax[sim].set(xlabel=r'$dk$', ylabel=r'$dt$')
        ax[sim].set_title('Simulation ' + str(sim+1) + ': ' + fft_titles[col])
    plt.savefig(field_plots+'_col'+str(col)+plot_fft_sim_suffix+'.png')
    return

def plot_real_space_slices(col,i,j,t):
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4))
    dx = range(nLat)
    for sim in range(nSims):
        ii = i
        slices = all_real_space[sim][col][0:nTime] #field amplitudes at each timeslice [128,512]
        for slice in slices[i:j:t]:
            ax[sim].plot(dx, slice, label=ii)
            ii = ii + t
        labelLines(ax[sim].get_lines(),xvals=(0, hLat),align=False,fontsize=16)
        ax[sim].set(xlabel=r'$dx$', ylabel = titles[col])
        ax[sim].set_title('Simulation '+str(sim+1))
        ax[sim].grid()
    plt.savefig(field_plots+'slices_i'+str(i)+'_to_j'+str(j)+'_every_t'+str(t)+'_col'+str(col)+plot_sim_suffix+'.png')
    return

def plot_specral_space_slices(col,i,j,t):
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4))
    dk = list(range(nyq))
    for sim in range(nSims):
        ii = i
        tableau = all_spectral_space[sim][col] #field amplitudes at each timeslice [128,512]
        pspec = [[abs(i*np.conj(i)) for i in tableau[j]] for j in range(nTime)]
        for slice in pspec[i:j:t]:
            ax[sim].plot(dk,slice,label=ii)
            ii = ii + t
        labelLines(ax[sim].get_lines(),xvals=(0,nyq),align=False,fontsize=16)
        ax[sim].set(xlabel=r'$dk$',ylabel = fft_titles[col])
        ax[sim].set_title('Simulation '+str(sim+1))
        ax[sim].grid()
    plt.savefig(field_plots+'slices_i'+str(i)+'_to_j'+str(j)+'_every_t'+str(t)+'_col'+str(col)+plot_fft_sim_suffix+'.png')
    return

def add_mask(col, threshold):
    masked_field = []
    if col != 2:
        print('Peak counting only works for smoothened fields. Choose different column.')
    for sim in range(nSims):
        mask = np.zeros((nTime,nLat))
        smoothened_field = all_real_space[sim][col] #open tableau of smoothened field values for each simulation
        for i in range(nLat):
            for j in range(nTime):
                if smoothened_field[j,i] > threshold:
                    mask[j,i] = 1
        masked_field.append(smoothened_field * mask)
    return(masked_field)
    
def plot_masked_field(col, threshold):
    masked_field = add_mask(col, threshold)
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4))
    for sim in range(nSims):
        im = ax[sim].imshow(masked_field[sim], aspect='auto', interpolation='none', origin='lower')     
        plt.colorbar(im, ax = ax[sim])
        ax[sim].set(xlabel=r'$dx$',ylabel=r'$dt$')
        ax[sim].set_title('Simulation '+str(sim+1)+' with Mask: '+titles[2])
    plt.savefig(field_plots + 'plot_masked_field'+'_col'+str(col)+plot_sim_suffix+'.png')
    return

def plot_masked_slices(col, threshold, i, j, t):
    fig,ax = plt.subplots(1,nSims,figsize=(4*nSims, 4))
    dx = list(range(nLat))
    masked_field = add_mask(col, threshold)
    for sim in range(nSims):
        ii = i
        smoothened_field_slices = masked_field[sim][0:nTime]
        for slice in smoothened_field_slices[i:j:t]:
            ax[sim].plot(dx, slice, label=ii)
            ii = ii + t
        labelLines(ax[sim].get_lines(),xvals=(0, hLat),align=False,fontsize=16)
        ax[sim].set(xlabel=r'$dx$',ylabel=titles[2])
        ax[sim].set_ylim(bottom=threshold)
        ax[sim].set_title('Simulation '+str(sim+1))
        ax[sim].grid()
    plt.savefig(field_plots + 'plot_masked_slices_i'+str(i)+'_to_j'+str(j)+'_every_t'+str(t)+'_col'+str(col)+plot_sim_suffix+'.png')  
    plt.show()
    return

############################################################
""" HELPER FUNCTIONS """
############################################################
    
def extract_data(filename, col):
    infile = open(filename,'r')
    lines = infile.readlines()
    field_values = [float(line.split()[col]) for line in lines[2:]]
    infile.close()
    return field_values

def complex_converter(txt):
    ii, jj = np.safe_eval(txt)
    return np.complex(ii, jj)

def extract_fft_data(filename, col):
    infile = open(filename,'r')
    lines = infile.readlines()
    str_values = [line.split()[col] for line in lines[2:]]
    complex_values = [complex_converter(str_values[i]) for i in range(len(str_values))]
    infile.close()
    return complex_values
    
def sim_location(sim):
    """ sim = simulation number """
    return './sims/t'+str(nTime)+'_x'+str(nLat)+'_sim'+str(sim)+'_fields.dat'

def sim_suffix(sim):
    return '_for_t'+str(nTime)+'_x'+str(nLat)+'_sim'+str(sim)+'_fields'

def fft_sim_location(sim):
    """ sim = simulation number """
    return './sims/t'+str(nTime)+'_x'+str(nLat)+'_sim'+str(sim)+'_fft_fields.dat'

def fft_sim_suffix(sim):
    return '_for_t'+str(nTime)+'_x'+str(nLat)+'_sim'+str(sim)+'_fft_fields'

def all_real_space_data():
    all_rsp_data = []
    for sim in range(nSims):
        sims_tableau = []
        for col in range(nCols):
            simulation = extract_data(sim_location(sim), col)
            simulation = np.reshape(simulation,(nTime, nLat))
            sims_tableau.append(simulation)
        all_rsp_data.append(sims_tableau)
#    print('Data format: ', np.shape(all_rsp_data),' and corresponds to: [simulation,variable, time slice, amplitude]')
#    print('Variables in order: field, time derivative, smoothened field, smoothened field momentum')
    return(all_rsp_data)

#####################################################################
# Careful. I am not considering the 0th mode!
def all_spectral_space_data():
    all_ssp_data = []
    for sim in range(nSims):
        sims_tableau = []
        for col in range(nCols):
            simulation = extract_fft_data(fft_sim_location(sim), col)
            simulation = np.reshape(simulation,(nTime, nyq))
            simulation = simulation[0:nTime, 1:nyq]
            sims_tableau.append(simulation)
        all_ssp_data.append(sims_tableau)
#    print('Data format: ', np.shape(all_ssp_data),' and corresponds to: [simulation, variable, frequency, amplitude]')
#    print('Variables in order: fft field, fft field momentum, fft smoothened field, fft smoothened field momentum')
    return(all_ssp_data)
    
all_real_space = all_real_space_data()
all_spectral_space = all_spectral_space_data()