#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:51:40 2021

@author: lukestew
"""

import numpy as np
import math
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

import lmfit
from lmfit import Model, minimize, fit_report, Parameters


def gwfreq(iM,iT,iT0):
    """
    
    Parameters
    ----------
    iM : float
        chirp mass
    iT : np.array
        array of times t (including merger time)
    iT0 : float
        merger time

    Returns
    -------
    w(t): angular velocity function as binary merger occurs
        
    """
    
    const = (948.5)*np.power((1./iM),5./8.)
    output = const*np.power(np.maximum((iT0-iT),3e-2),-3./8.) # we can max it out above 500 Hz-ish
    return output


def osc(t, Mc, t0, C, phi):
    """

    Parameters
    ----------
    t : np.array
        array of times t
    Mc : float
        chirp mass
    t0 : float
        merger time
    C : float
        scalar coefficient
    phi : float
        phase

    Returns
    -------
    array: rudimentary fit ansatz with inspiral, collision,
           decaying exponential ringdown
        
    """
    
    vals = []
    for time in t:
        if time <= t0:
            vals.append(C * (Mc * gwfreq(Mc, time, t0))**(10/3) * np.cos(gwfreq(Mc, time, t0) * (time-t0) + phi))
        else:
            vals.append(np.exp(-100*(time-t0)) * C * (Mc * gwfreq(Mc, time, t0))**(10/3) * np.cos(gwfreq(Mc, time, t0) * (time-t0) + phi))

    return np.asarray(vals)


def osc_dif(params, x, data, eps):
    """
    
    Parameters
    ----------
    params : lmfit.parameter class instance
        Initializes and provides guesses for all fit params
    x : np.array
        Array of times containing merger
    data : np.array
        Array of whitened, bp'd, zoomed strain data
    eps : np.array

    Returns
    -------
    np.array
        Residual of fit - data given input fit parameters

    """
    
    iM=params["Mc"]
    iT0=params["t0"]
    norm=params["C"]
    phi=params["phi"]
    val=osc(x, iM, iT0, norm, phi)
    
    return (val-data)/eps



# Set parameters
# ------------------------------------------------
fn = 'data/GW151012/H-H1_GWOSC_4KHZ_R1-1128676853-4096.hdf5' # data file
treport = 1128678900.4
tevent = 1128678900.42 # Mon Sep 14 09:50:45 GMT 2015
evtname = 'GW150914' # event name

detector = 'L1' # detector: L1 or H1



# Load LIGO data
#----------------------------------------------------------------
strain = TimeSeries.read(fn, format='hdf5.losc')
center = int(tevent)
strain = strain.crop(center-16, center+16)



# Obtain the power spectrum density PSD / ASD
#----------------------------------------------------------------
asd = strain.asd(fftlength=8)



# Whitening data
#----------------------------------------------------------------
white_data = strain.whiten()

# print(white_data)



# Bandpass filtering
#----------------------------------------------------------------
bandpass_low = 30
bandpass_high = 400

white_data_bp = white_data.bandpass(bandpass_low, bandpass_high)

# plt.figure()
# white_data_bp.plot()
# plt.ylabel('strain (whitened + band-pass)')

# plt.figure()
# white_data_bp.plot()
# plt.ylabel('strain zoomed (whitened + band-pass)')
# plt.xlim(tevent-1, tevent+1)

# white_data_bp.plot()


#----------------------------------------------------------------
# q-transform
#----------------------------------------------------------------

dt = 1  #-- Set width of q-transform plot, in seconds
hq = white_data_bp.q_transform(outseg=(tevent-dt, tevent+dt))

plt.clf()
fig = hq.plot()
ax = fig.gca()
fig.colorbar(label="Normalised energy")
ax.grid(False)
plt.xlim(tevent-0.5, tevent+0.4)
plt.yscale('log')
plt.ylabel('Frequency (Hz)')



# Plot the fit ansatz
#----------------------------------------------------------------
times = np.linspace(-0.1, 0.3, 1000)
freq = osc(times, 30, 0.18, 1, 0.0)
omega = gwfreq(30, times, .18)

# plt.figure(figsize=(12, 4))
# plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
# plt.plot(times, freq)
# plt.xlabel('Time (s) since '+str(tevent))
# plt.ylabel('strain')



# Fit
#----------------------------------------------------------------

sample_times = white_data_bp.times.value
sample_data = white_data_bp.value
indxt = np.where((sample_times >= (tevent-0.17)) & (sample_times < (tevent+0.13)))
x = sample_times[indxt]
x = x-x[0]
white_data_bp_zoom = sample_data[indxt]

pulse_ind = np.where((sample_times >= (tevent-0.001)) & (sample_times < (tevent+0.001)))
pulse_time = np.average(sample_times[pulse_ind] - sample_times[indxt][0])

report_time_ind = np.where((sample_times >= (treport-0.001)) & (sample_times < (treport+0.001)))
report_time = np.average(sample_times[report_time_ind] - sample_times[indxt][0])


plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(x, white_data_bp_zoom)
plt.xlabel('Time (s)')
plt.ylabel('strain (whitened + band-pass)')

model = lmfit.Model(osc)
p = model.make_params()
p['Mc'].set(20)     # Mass guess
p['t0'].set(0.18)  # By construction we put the merger in the center
p['C'].set(1e-12)      # normalization guess
p['phi'].set(0)    # Phase guess
unc = np.full(len(white_data_bp_zoom),20)
out = minimize(osc_dif, params=p, args=(x, white_data_bp_zoom, unc))
print("----------------------------------")
print("tevent:", tevent)
print('LIGO reported time:', report_time)
print('t0: {:f}'.format(out.params['t0'].value))
print('t0 sigma: {:f}'.format(out.params['t0'].stderr))
print(fit_report(out))
Mc = out.params['Mc']
t0 = out.params['t0']
C = out.params['C']
phi = out.params['phi']

plt.plot(x, osc(x, Mc, t0, C, phi))

plt.plot(x, model.eval(params=out.params,t=x),'r',label='best fit')
plt.plot(pulse_time * np.ones(100), np.linspace(-4,4,100), color='black')


fit_function = np.array(model.eval(params=out.params,t=x))


# plt.figure()
# plt.plot(x, (white_data_bp_zoom - fit_function))


remaining_signal = (white_data_bp_zoom - fit_function) / (.3)
print('remaining = ', np.std(remaining_signal))
print(np.where(remaining_signal == max(remaining_signal)))
print(remaining_signal[223])
print(white_data_bp_zoom[223])
print(fit_function[223])

# plt.savefig("good strain fit", dpi=300)
# plt.show()








