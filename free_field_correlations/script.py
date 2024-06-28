#import matplotlib
#matplotlib.use('Agg')

import analysis as analysis

t_ref = 0
x_ref = 0
thresh = 0
b_size = 1

th_pspec()
spacelike_fft_th_pspec()
timelike_fft_th_pspec()

######################################################################

col=0

######################################################################

space_pspec_th_and_data(col)
time_pspec_th_and_data(col)
spacelike_two_point_function(col, x_ref)
timelike_two_point_function(col, t_ref)

######################################################################

col=2

######################################################################

space_pspec_th_and_data(col)
time_pspec_th_and_data(col)
spacelike_two_point_function(col, x_ref)
timelike_two_point_function(col, t_ref)

peak_peak_correlation(col, thresh, b_size)