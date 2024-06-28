"""
analysis.position_position_equal_amplitude_at_time(col, timeslice, accuracy)
analysis.time_time_equal_amplitude_at_site(col, spaceslice, accuracy)
analysis.position_position_equal_amplitude_at_sim(col, sim,accuracy)
analysis.time_time_equal_amplitude_at_sim(col, sim, accuracy)
analysis.position_position_equal_amplitude_overall(col, accuracy)
analysis.time_time_equal_amplitude_overall(col, accuracy)
"""

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

#%matplotlib qt
#%matplotlib inline

locationUbuntu = '/mnt/c/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/improved_Scalar/'
locationPythonOG = 'C:/Users/dpirvu/Documents/PSI/Project/1d-ScalarOG/improved_Scalar/'
correl_plots = 'plots/correlations/'
sim_location = 'sims/sigma10_t250x250_'
sim_suffix = '_for_sigma10_t250x250'
    
nLat = 250
nTime = 250
nSims = 10
nCols = 3
sigma = 10
nyq = int(nLat/2)+1
hLat = int(nLat/2)
dk = 2.*np.pi/nLat
Vol = nTime*nLat
m2eff = 1
m = np.sqrt(m2eff)
dx = 1.
dt = 1./8.

recombination_time = [t for t in range(nTime) if t%int(m**(-1))==0]
titles = [r'Field $\phi(x)$',r'$\partial_t \phi(x)$',r'Smoothened $\phi(x)$']
fft_titles = [r'$\phi(k)$',r'$\partial_t \phi(k)$',r'Smoothened $\phi(k)$']

############################################################
""" Amplitude correlation from real space data"""
############################################################

def position_equal_amplitude_overall(col, accuracy):
    """Computes all distances between fields of same amplitude at different locations across all simulations and timeslices."""
    distances = [position_distance_at_equal_amplitude(col, sim, timeslice, accuracy) for sim in range(nSims) for timeslice in range(nTime)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    for i in distances:
        if i > nLat/2:
            i = nLat-i # takes shortest distance between points on the circular lattice
    return distances

def position_equal_amplitude_at_sim(col, sim, accuracy):
    """Computes all distances between fields of same amplitude at different locations averaged over timeslice, per simulation."""
    distances = [position_distance_at_equal_amplitude(col, sim, timeslice, accuracy) for timeslice in range(nTime)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    for i in distances:
        if i > nLat/2:
            i = nLat-i # takes shortest distance between points on the circular lattice
    return distances

def position_equal_amplitude_at_timeslice(col, timeslice, accuracy):
    """Computes all distances between fields of same amplitude at different locations averaged over simulations, at fixed timeslice."""
    distances = [position_distance_at_equal_amplitude(col, sim, timeslice, accuracy) for sim in range(nSims)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    for i in distances:
        if i > nLat/2:
            i = nLat-i # takes shortest distance between points on the circular lattice
    return distances

def position_distance_at_equal_amplitude(col, sim, timeslice, accuracy):
    """For a set time and simulation, finds distance between two point with the same field amplitude."""
    field = all_real[sim][col][timeslice]
    matrix = [[None for x1 in range(nLat)] for x2 in range(nLat)]
    for x1 in range(nLat):
        for x2 in range(nLat):
            matrix[x1][x2] = round(field[x1]-field[x2], accuracy)
    zeros = np.transpose(np.nonzero(np.asarray(matrix) == 0)) # takes coordinates x1 and x2 where \phi(x1)-\phi(x2)==0
    dist = [zeros[i][0] - zeros[i][1] for i in range(len(zeros))] # computes distance on lattice between said points
    dist = [i for i in dist if i > 0] #ignores distance ==0 i.e. any \phi(x) is equal to itself
    return dist

##############################################################

def time_equal_amplitude_overall(col, accuracy):
    """Computes all distances between fields of same amplitude at different times across all simulations and position on the lattice = spaceslice."""
    distances = [time_distance_at_equal_amplitude_at_sim_at_site(col, sim, spaceslice, accuracy) for sim in range(nSims) for spaceslice in range(nLat)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    # distance in time does not require periodic boundary conditions
    return distances

def time_equal_amplitude_at_sim(col, sim, accuracy):
    """Computes all distances between fields of same amplitude at different times averaged over all lattice sites = spaceslices, per simulation."""
    distances = [time_distance_at_equal_amplitude_at_sim_at_site(col, sim, spaceslice, accuracy) for spaceslice in range(nLat)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    return distances

def time_equal_amplitude_at_site(col, spaceslice, accuracy):
    """Computes all distances between fields of same amplitude at different times averaged over simulations, at fixed lattice site = spaceslice."""
    distances = [time_distance_at_equal_amplitude_at_sim_at_site(col, sim, spaceslice, accuracy) for sim in range(nSims)]
    distances = [item for sublist in distances for item in sublist] # flattens list of lists
    return distances

def time_distance_at_equal_amplitude_at_sim_at_site(col, sim, spaceslice, accuracy):
    """For a set pair of lattice sites and simulation, finds distance in time between two instances with the same field amplitude."""
    field = [all_real[sim][col][i][spaceslice] for i in range(nTime)]
    matrix = [[None for y in range(nTime)] for x in range(nTime)]
    for t1 in range(nTime):
        for t2 in range(nTime):
            matrix[t1][t2] = round(field[t1] - field[t2], accuracy)
    zeros = np.transpose(np.nonzero(np.asarray(matrix) == 0))
    dist = [zeros[i][0] - zeros[i][1] for i in range(len(zeros))]
    dist = [i for i in dist if i > 0]
    return dist



######################################################################

def position_position_equal_amplitude_at_time(col, timeslice, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances on the lattice, across all simulations, at fixed time."""
    plt.figure(figsize=(4,4))
    data = position_equal_amplitude_at_timeslice(col, timeslice, accuracy)
    freqs = [data.count(R) for R in range(2,hLat)]
    bins = [i for i in range(2,int(hLat/2))]
    plt.hist(freqs, bins, facecolor='g', alpha=0.75)
    plt.xlabel(r'$\Delta R$')
    plt.ylabel('Fraction')
    plt.title('Equal field amplitude at time = '+str(timeslice))
    plt.grid()
    plt.savefig(correl_plots + 'position_position_equal_amplitude_at_time_'+str(timeslice)+sim_suffix+'.png')
    return 

def time_time_equal_amplitude_at_site(col, spaceslice, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances in time, across all simulations, at fixed lattice site."""
    plt.figure(figsize=(4,4))
    data = time_equal_amplitude_at_site(col, spaceslice, accuracy)
    freqs = [data.count(R) for R in range(2,nTime)]
    bins = [i for i in range(2,int(nTime/2))]
    plt.hist(freqs, bins, facecolor='b', alpha=0.75)
    [plt.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Fraction')
    plt.title('Equal field amplitude at x = '+str(spaceslice))
    plt.grid()
    plt.savefig(correl_plots + 'time_time_equal_amplitude_at_site_'+str(spaceslice)+sim_suffix+'.png')
    return

def position_position_equal_amplitude_at_sim(col, sim, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances on the lattice, across time."""
    plt.figure(figsize=(4,4))
    bins = [i for i in range(2,int(hLat/2))]
    data = position_equal_amplitude_at_sim(col, sim, accuracy)
    freqs = [data.count(R) for R in range(2,hLat)]
    plt.hist(freqs, bins, facecolor='g', alpha=0.75)
    plt.xlabel(r'$\Delta R$')
    plt.grid()
    plt.title('Equal field amplitude at sim = '+str(sim+1))
    plt.savefig(correl_plots + 'position_position_equal_amplitude_at_sim_' +str(sim+1)+sim_suffix+'.png')
    return 

def time_time_equal_amplitude_at_sim(col, sim, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances in time, across lattice."""
    plt.figure(figsize=(4,4))
    bins = [i for i in range(2,int(nTime/2))]
    data = time_equal_amplitude_at_sim(col, sim, accuracy)
    freqs = [data.count(R) for R in range(2,nTime)]
    plt.hist(freqs, bins, facecolor='b', alpha=0.75)
    [plt.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    plt.xlabel(r'$\Delta t$')
    plt.grid()
    plt.title('Equal field amplitude at sim = '+str(sim+1))
    plt.savefig(correl_plots + 'time_time_equal_amplitude_at_sim_' +str(sim+1)+sim_suffix+'.png')
    return

def position_position_equal_amplitude_overall(col, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances on the lattice, across all available data."""
    plt.figure(figsize=(4,4))
    data = position_equal_amplitude_overall(col, accuracy)
    freqs = [data.count(R) for R in range(2,hLat)]
    bins = [i for i in range(2,int(hLat/2))]
    plt.hist(freqs, bins, facecolor='g', alpha=0.75)
    plt.xlabel(r'$\Delta R$')
    plt.ylabel('Fraction')
    plt.title('Equal field amplitude')
    plt.grid()
    plt.savefig(correl_plots + 'position_position_equal_amplitude_overall_'+sim_suffix+'.png')
    return 

def time_time_equal_amplitude_overall(col, accuracy):
    """Plots histogram of occurences of same field aplitude at all possible distances in time, across all available data."""
    plt.figure(figsize=(4,4))
    data = time_equal_amplitude_overall(col, accuracy)
    freqs = [data.count(R) for R in range(2,nTime)]
    bins = [i for i in range(2,int(nTime/2))]
    plt.hist(freqs, bins, facecolor='b', alpha=0.75)
    [plt.axvline(t, color='silver', linestyle='dotted') for t in recombination_time]
    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Fraction')
    plt.title('Equal field amplitude')
    plt.grid()
    plt.savefig(correl_plots + 'time_time_equal_amplitude_overall_'+sim_suffix+'.png')
    return

""""""""""""""""""""
" Helper Functions "
""""""""""""""""""""
def complex_converter(txt):
    ii, jj = np.safe_eval(txt)
    return np.complex(ii, jj)

def extract_header(filename):
    infile = open(filename,'r')
    print(infile.readline(),infile.readline())
    infile.close()
    return

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
    
def difference_matrix(a):
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()
    
def sub(lst):
    sub = lst[0]
    for i in range(len(lst)-1):
        sub = sub - lst[i+1]
    return sub

