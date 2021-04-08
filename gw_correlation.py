import numpy as np
import math
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import sys
import lmfit
from lmfit import Model, minimize, fit_report, Parameters

### certain functions written by luke

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


def raw_to_bandpass(strain_file, t_event, detector, bp_lo, bp_hi):
    strain = TimeSeries.read(strain_file, format='hdf5.losc')
    center = int(tevent)
    strain = strain.crop(center-16, center+16)
    white_data = strain.whiten()
    white_data_bp = white_data.bandpass(bp_lo, bp_hi)

    return white_data_bp

################ new functions for correlation ########################

def reduced_chi_square(data, template, dof):
    chi_square = np.sum((data-template)**2/(0.1*data)**2)
    reduced = chi_square/dof
    return reduced

def running_fit_check(times, data, step):
    if len(times) != len(data):
        raise IndexError

    times_for_template = np.arange(-0.3, 0.1, 1000)

    model = lmfit.Model(osc)
    p = model.make_params()
    p['Mc'].set(20)    # Mass guess
    p['t0'].set(0)  # By construction we put the merger in the center
    p['C'].set(1)      # normalization guess
    p['phi'].set(0)    # Phase guess

    dt = times[1] - times[0]
    t0 = times[0]

    num_in_range = math.ceil(0.4/dt)

    chi_squared_list = []

    for i in range(0, len(times)-num_in_range, step):
        data_to_fit = data[i:i+num_in_range]
        times_to_fit = times[i:i+num_in_range]

        unc = np.full(len(data_to_fit),20)
        out = minimize(osc_dif, params=p, args=(times_to_fit, data_to_fit, unc))
        
        template = model.eval(params=out.params,t=times_for_template)

        single_chi = reduced_chi_square(data_to_fit, template, len(data_to_fit)-4)

        chi_squared_list.append(single_chi)

        sys.stdout.write("Progress: " + str(i) + "/" + str(len(times)-num_in_range))
        sys.stdout.flush()
    
    return chi_squared_list, np.linspace(0, len(chi_squared_list)*dt*step, len(chi_squared_list))

#######################################################

### reading in files

fn1 = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'
fn2 = 'data/L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'

tevent = 1126259462.422 # Mon Sep 14 09:50:45 GMT 2015
evtname = 'GW150914' # event name

#######################################################

h1_strain = raw_to_bandpass(fn1, tevent, "H1", 30, 350)
l1_strain = raw_to_bandpass(fn2, tevent, "L1", 30, 350)

### cropped to just around the event

h1_strain_c = h1_strain.crop(tevent-0.17, tevent+0.13)
l1_strain_c = l1_strain.crop(tevent-0.17, tevent+0.13)

### time step from h1, not sure if this is the same for l1 or not

dt = 0.000244140625
t0 = 1126259446.0

# plt.plot(np.arange(0, len(correlation)*dt, dt), -1*correlation)
# # plt.plot(h1_strain)
# # plt.plot(l1_strain)
# plt.xlabel("Time (s)")
# plt.ylabel("Strain$^2$")
# plt.title("Correlation between Hanford and Livingston")
# plt.show()

#####################################################
####### sliding template correlation plots ##########

times = np.linspace(-0.3, 0.1, 1000)
strain = osc(times, 17, 0, 0.1, 0.0)

# correlation = np.correlate(l1_strain, h1_strain, mode='full')
correlation1 = np.correlate(h1_strain, strain, mode='full')
correlation2 = np.correlate(l1_strain, strain, mode='full')
correlation = np.correlate(correlation1, correlation2, mode='full')

# fig, axs = plt.subplots(2)

# axs[0].plot(np.arange(0, len(correlation1)*dt, dt), -1*correlation1, label="Hanford", color='r')
# axs[0].title.set_text("Hanford vs. Template")
# axs[0].set_ylabel("Correlation")
# axs[1].plot(np.arange(0, len(correlation2)*dt, dt), -1*correlation2, label="Livingston", color='b')
# axs[1].title.set_text("Livingspton vs. Template")
# axs[1].set_ylabel("Correlation")
# # axs[2].plot(np.arange(0, len(correlation)*dt, dt), -1*correlation, color='k')
# # axs[2].title.set_text("Hanford vs. Livingston")
# # axs[2].set_ylabel("Correlation")
# #axs[0].vlines(x=0.13, ymin=-4000, ymax=5000, linestyles='--', color='k', label="Event Time")
# plt.tight_layout()
# plt.xlabel("Time (s)")

# h1_strain_c.plot()
# plt.plot(times + tevent, strain)

# plt.show()

#########################################################
################# sliding fit method ####################

h1_times = np.arange(t0, t0+len(h1_strain)*dt, dt)
h1_times2 = np.arange(0, len(h1_strain)*dt, dt)
print(len(h1_times))

#plt.plot(h1_times2[65000:69000], h1_strain[65000:69000])

chi_squared_list, x_list = running_fit_check(h1_times2[60000:80000], h1_strain[60000:80000], 100)
print(chi_squared_list)

plt.plot(h1_times2[60000:80000], h1_strain[60000:80000])
plt.plot(x_list+h1_times2[60000], [(i**-1)*10**6 for i in chi_squared_list])
plt.show()