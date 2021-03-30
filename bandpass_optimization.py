#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:44:28 2021

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
    array : rudimentary fit ansatz with inspiral, collision,
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
    array : Residual of fit - data given input fit parameters

    """
    
    iM=params["Mc"]
    iT0=params["t0"]
    norm=params["C"]
    phi=params["phi"]
    val=osc(x, iM, iT0, norm, phi)
    
    return (val-data)/eps


def bandpass_filtering(bp_lo, bp_hi, white_data, tevent):
    """
    
    Parameters
    ----------
    bp_lo : float
        Lower bandpass [Hz]
    bp_hi : float
        Upper bandpass [Hz]
    white_data : gwpy.timeseries instance
        Whitened data
    t_event : float
        Merger time
    
    Returns
    -------
    MinimizerResult : 
        Values of fitted parameters
    
    """

    white_data_bp = white_data.bandpass(bp_lo, bp_hi)
    
    sample_times = white_data_bp.times.value
    sample_data = white_data_bp.value
    indxt = np.where((sample_times >= (tevent-0.17)) & (sample_times < (tevent+0.13)))
    x = sample_times[indxt]
    x = x-x[0]
    
    white_data_bp_zoom = sample_data[indxt]
    
    model = lmfit.Model(osc)
    p = model.make_params()
    p['Mc'].set(20)     # Mass guess
    p['t0'].set(0.18)  # By construction we put the merger in the center
    p['C'].set(1)      # normalization guess
    p['phi'].set(0)    # Phase guess
    unc = np.full(len(white_data_bp_zoom),20)
    
    out = minimize(osc_dif, params=p, args=(x, white_data_bp_zoom, unc))
    
    fit_function = np.array(model.eval(params=out.params,t=x))
    
    chi_square = 0

    for i in range(len(white_data_bp_zoom)):
        chi_square += (white_data_bp_zoom[i] - fit_function[i])**2 / (abs(white_data_bp_zoom[i]))
    
    for i in range(len(white_data_bp_zoom)):
        chi_square += (white_data_bp_zoom[i] - fit_function[i])**2 / (abs(white_data_bp_zoom[i]))

    redchi = chi_square / (len(white_data_bp_zoom) - 4)
    
    r_sq = 1 - out.chisqr / np.var(white_data_bp_zoom)
    
    print('bandpass: {:.0f} - {:.0f}; Mc: {:.3f}; r_sq = {:.6f}'.format(bp_lo, bp_hi, out.params['Mc'].value, r_sq))
    
    return out, redchi



# Set parameters
#----------------------------------------------------------------
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


# Iterate upper bandpass limit
#----------------------------------------------------------------

result_dict = {'bp': [], 'Mc':[], 't0':[], 'C':[], 'phi':[], 'redchi':[]}

for bp in np.linspace(80,600, 20):
    
    result_dict['bp'].append(bp)
    
    out, redchi = bandpass_filtering(30, bp, white_data, tevent)
    
    for param in out.params:
        
        # append fitted parameters to lists in result_dict
        result_dict[param].append(out.params[param].value)
    
    result_dict['redchi'].append(redchi)


# Iterate lower bandpass limit
#----------------------------------------------------------------

# result_dict = {'bp': [], 'Mc':[], 't0':[], 'C':[], 'phi':[], 'redchi':[]}

# for bp in np.linspace(10, 100, 45):
    
#     result_dict['bp'].append(bp)
    
#     out, redchi = bandpass_filtering(bp, 350, white_data, tevent)
    
#     for param in out.params:
        
#         # append fitted parameters to lists in result_dict
#         result_dict[param].append(out.params[param].value)  
    
#     result_dict['redchi'].append(redchi)
        

lower_bp = result_dict['bp']
redchi = result_dict['redchi']
Mc = result_dict['Mc']

plt.scatter(lower_bp, redchi, color='k')
plt.xlabel('Lower bandpass filter limit (__ - 350) $[Hz]$')
plt.ylabel('Reduced $\chi^2$')

for i, bp in enumerate(lower_bp):
    plt.text(bp, redchi[i], round(Mc[i],1), fontsize=8)

plt.tight_layout()
plt.savefig("Lower bandpass chisq sensitivity-2", dpi=300)
plt.show()











