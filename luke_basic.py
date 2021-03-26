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
            vals.append(C*1e-12 * (Mc * gwfreq(Mc, time, t0))**(10/3) * np.cos(gwfreq(Mc, time, t0) * (time-t0) + phi))
        else:
            vals.append(np.exp(-100*(time-t0)) * C*1e-12 * (Mc * gwfreq(Mc, time, t0))**(10/3) * np.cos(gwfreq(Mc, time, t0) * (time-t0) + phi))

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
fn = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5' # data file
tevent = 1126259462.422 # Mon Sep 14 09:50:45 GMT 2015
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



# Bandpass filtering
#----------------------------------------------------------------
bandpass_low = 30
bandpass_high = 350

white_data_bp = white_data.bandpass(bandpass_low, bandpass_high)

plt.figure()
white_data_bp.plot()
plt.ylabel('strain (whitened + band-pass)')

plt.figure()
white_data_bp.plot()
plt.ylabel('strain zoomed (whitened + band-pass)')
plt.xlim(tevent-.17, tevent+.13)



# Plot the fit ansatz
#----------------------------------------------------------------
times = np.linspace(-0.1, 0.3, 1000)
freq = osc(times, 30, 0.18, 1, 0.0)
omega = gwfreq(30, times, .18)

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(times, freq)
plt.xlabel('Time (s) since '+str(tevent))
plt.ylabel('strain')



# Fit
#----------------------------------------------------------------

sample_times = white_data_bp.times.value
sample_data = white_data_bp.value
indxt = np.where((sample_times >= (tevent-0.17)) & (sample_times < (tevent+0.13)))
x = sample_times[indxt]
x = x-x[0]
white_data_bp_zoom = sample_data[indxt]

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(x, white_data_bp_zoom)
plt.xlabel('Time (s)')
plt.ylabel('strain (whitened + band-pass)')

model = lmfit.Model(osc)
p = model.make_params()
p['Mc'].set(20)     # Mass guess
p['t0'].set(0.18)  # By construction we put the merger in the center
p['C'].set(1)      # normalization guess
p['phi'].set(0)    # Phase guess
unc = np.full(len(white_data_bp_zoom),20)
print(unc)
out = minimize(osc_dif, params=p, args=(x, white_data_bp_zoom, unc))
print(fit_report(out))
plt.plot(x, model.eval(params=out.params,t=x),'r',label='best fit')
plt.show()

for i in out.params:
    print(i)





