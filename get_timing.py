# %%
import numpy as np
import pandas as pd
import time
import multiprocessing
import pathlib
import scipy
import kdetools
from scipy.integrate import quad
from helpful_functions import *
# %%

########  DEFINE GLOBAL UNITS & CONSTANTS

GeV = 1
mp = 0.938*GeV
gram = 5.62e23*GeV
km = 1
cm = 1/(1e5) * km

sec=1
c = 299792.458*km/sec

######## DEFINE EARTH COMPOSITION PARAMETERS

Rearth = 6371*km
Rcore = 3480*km
Rmantle = 6346*km

####### DEFINE TARGET IN DETECTION EXPERIMENT
target_name = "Xenon"

######## DEFINE FUNCTIONS TO CALCULATE TIMING OF SCATTERS

# returns: expected constituent spread size, weighted average over entry angles
# inputs, mx, sigmand
def get_spreadsize(mx, sigmand):
    data = np.load("DataProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand), allow_pickle=True)

    spreadsizes = np.float64(data[:,3])

    return np.mean(spreadsizes)

# returns: number density of target atom in noble liquid detector (atom/cm^3)
# input: name of element ("Xenon", "Argon")
def nAdetector(element):
    nAr = 1.4 * 6.022e23/40 # target atoms per cm^3
    nXe = 3.52 * 6.022e23/131 # target atoms per cm^3
    if element == "Xenon":
        return  nXe
    elif element == "Argon":
        return nAr
    return 0

# returns: expected flux of DM composites through the earth (sec^-1)
# input: constituent mass (mx) and constituent number (Nd)
def num_composites_earthpersec(Nd, mx):
    rhodm = 0.3*GeV/cm**3 # in m/km^3
    Md = Nd*mx*GeV
    veldm = 1e-3*c # km/s
    flux = rhodm*veldm/Md # km^(-2)s^-1
    return flux*(np.pi*Rearth**2)

# returns: expected constituent speed, weighted average over entry angles
# inputs, mx, sigmand
def get_velocity(mx, sigmand):
    data = np.load("DataProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand), allow_pickle=True)
    vs = np.float64(data[:,-1])
    return np.mean(vs)

# returns: DM constituent-nucleus cross-section with target nucleus, cm^2
# input: index of target nucleus, DM mass in GeV,  DM constituent-nucleon cross-section in cm^2, and DM velocity in km/s.
def sigmaAddetector(element, sigmand, mx, vel): # cm^2
    if element == "Xenon":
        A = 131
    elif element == "Argon":
        A = 40
    mA = A*mp

    Ermax = 2*mu(mx, mA)**2*(vel)**2/mA

    func_dsigmaAd = lambda Er: A**2 * sigmand * (mu(mA, mx)/mu(mp, mx))**2 * helm2(np.sqrt(2*mA*Er), Rnuc(A), s=0.9/0.197)* helm2(np.sqrt(2*mA*Er), Rnuc(A), s=0.9/0.197)*np.heaviside(np.sqrt(2)*vel*mu(mx, mA)- np.sqrt(2*mA*Er), 1)

    sigmaAd = mA/(2*(mu(mA, mx)*(vel))**2)*quad(func_dsigmaAd, 0, Ermax, epsabs=1.e-20, epsrel=1.e-20, limit=50)[0]

    return sigmaAd

# returns: expected # of scatters within detecor
def ScattersPerCone(mx, sigma, Nd, Rconedata, target='Xenon'):
    Rcone = Rconedata/cm # km -> cm
    Adetector = 1*(100**2) # 1 m^2 -> cm^2
    l = 1 * 100 # cm

    constituentsindetector = min(Nd*Adetector/(np.pi*Rcone**2), Nd)
    vel = get_velocity(10**mx, 10**sigma)/c
    scatterperconstituent = nAdetector(target)*sigmaAddetector(target, 10**sigma, 10**mx, vel)*l
    return constituentsindetector*scatterperconstituent




# returns: what is the time delay between two successive scatters from the same constituent cloud,
#           for a range of constituent numbers (N_D)?
# input: data array as above.
def get_dt(data):
    # given a number of scatters per cone, what is dt between two scatters?
    mx = data[0]
    sigmand = data[1]
    Nds = [1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
    dts = []
    processes = []

    for Nd in Nds:

        rcone = get_spreadsize(mx, sigmand)
        num_scatters = ScattersPerCone(np.log10(mx), np.log10(sigmand), Nd, rcone)
        if num_scatters <= 1:
            dts.append(np.nan)
            processes.append(np.nan)

            continue

        print("== Run for mx = {} GeV, sigma = {} cm^2.".format(mx, sigmand))
        filesR = np.sort([str(f) for f in pathlib.Path().glob("DataProcessed/SpreadRs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        filesT = np.sort([str(f) for f in pathlib.Path().glob("DataProcessed/FinalTs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])

        filesV = np.sort([str(f) for f in pathlib.Path().glob("DataProcessed/FinalVs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
        ind = filesR[0].find("angle-")+len("angle-")
        angles = [f[ind:ind+15] for f in filesR]
        proc = 0

        mean_tdiff = 0
        numangles = len(filesR)

        for i in range(numangles):

            scatter_ts = []

            rs = np.load(filesR[i], allow_pickle=True)
            ts = np.load(filesT[i], allow_pickle=True)#*1e9
            vs = np.load(filesV[i], allow_pickle=True)

            tsnew = ts[np.where(np.isnan(ts) == False)]
            rs = rs[np.where(np.isnan(ts) == False)]
            ts = tsnew

            # print("T RANGE : {}/{}".format(i, numangles), mx, sigmand, angles[i], np.max(ts), np.min(ts), np.max(ts) - np.min(ts))

            z_score = np.abs(scipy.stats.zscore(np.log10(ts), nan_policy='propagate'))
            z_score = np.where(np.isnan(z_score), 0, z_score)

            nonoutliers = np.where(z_score < 3)[0]

            ts = ts[nonoutliers]
            rs = rs[nonoutliers]

            print("T RANGE:", np.max(ts), np.min(ts), np.max(ts) - np.min(ts))

            if np.max(ts) - np.min(ts) <= 1e-8:
                # we do not expect the dt to be larger than nanoseconds,
                # we do not plot this data point
                mean_tdiff +=(np.max(ts) - np.min(ts))/num_scatters
                print("NO POINT IN RUNNING", mx, sigmand, angles[i], (np.max(ts) - np.min(ts))/num_scatters)
                proc += 0
            else:
                numtrials=200
                data = np.stack((rs, ts), axis=1)
                dtoverr = 0

                if rcone > 1e-3:
                    try:
                        time_kde = kdetools.gaussian_kde(data.T)
                        print("WE ARE TRYING")

                        for j in range(numtrials):
                            rval = rcone*np.sqrt(np.random.rand())
                            tsample = time_kde.conditional_resample(1000, x_cond=np.array([rval]), dims_cond=[0]).ravel()
                            if np.max(tsample) - np.min(tsample) <= 1e-8:
                                dt =(np.max(tsample) - np.min(tsample))/num_scatters
                                print("TOO SMALL", dt)

                            else:
                                (mu, sigma) = scipy.stats.norm.fit(tsample)
                                twindow = 2*sigma
                                dt = twindow/num_scatters
                            # print(np.min(tsample), np.max(tsample), mu, (np.max(ts) - np.min(ts)), rval)
                            dtoverr += dt/numtrials
                    except:
                        print("WE ARE EXCEPTING")
                        dtoverr = 0
                        (mu, sigma) = scipy.stats.norm.fit(ts)
                        twindow = 2*sigma 

                        dtoverr = twindow/num_scatters
                        proc += 2

                else:
                    print("WE ARE EXCEPTING")
                    dtoverr = 0
                    (mu, sigma) = scipy.stats.norm.fit(ts)
                    twindow = 2*sigma 

                    dtoverr = twindow/num_scatters
                    proc += 2
                
                print("WE RAN IT",mx, sigmand, angles[i], np.max(ts) - np.min(ts), dtoverr)
                mean_tdiff += dtoverr
        print(mean_tdiff/numangles)
        dts.append(mean_tdiff/numangles)
        processes.append(proc)


    pd.to_pickle(np.array([Nds, dts, processes]), "DataProcessed/FinalDTs_mx-{}_sigma-{}.pkl".format(mx, sigmand))
    print("Done saving! -- ", dts)

    return 0


######## RUNNING & PARALLELIZING THE CODE

cpus = int(multiprocessing.cpu_count())
print("Working across {} CPUs.".format(cpus))
# mxs = np.append(10**np.linspace(2, 10, 20), [10**np.float64(10.421052631578947), 10**np.float64(10.842105263157894), 10**np.float64(11.263157894736842), 10**np.float64(11.68421052631579), 10**np.float64(12.105263157894736)])
# sigmas = 10**np.linspace(-41, -37, 20)
# mxs = [mxs[-6]]
args = []
mxs = [1e3]
sigmas = [10**np.linspace(-41, -37, 20)[-5]]

# get_dt([mxs[0], sigmas[5]])

for i in range(len(mxs)):
    for j in range(len(sigmas)):
        args.append([mxs[i], sigmas[j]])

start_time = time.time()

if __name__ == "__main__":
    
    # SET UP MULTIPROCESSING

    num_pool = cpus

    pool = multiprocessing.Pool(processes = num_pool, maxtasksperchild=2)
    
    doing_tasks = pool.map(get_dt, args)
    
    print("Processes are all done, in {} minutes!".format((time.time() - start_time)/60))

# %%
