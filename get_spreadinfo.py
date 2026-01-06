# %%
import numpy as np
import pandas as pd
import time
import multiprocessing
import pathlib
import scipy as sp
from helpful_functions import *

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

# element mass number
elements = np.array([16, 28, 27, 56, 40, 23, 39, 24, 48, 57, 59, 31, 32])

# percent by weight of each element in each Earth layer
crust = np.array([46.7, 27.7, 8.1, 5.1, 3.7, 2.8, 2.6, 2.1, 0.6, 0.0, 0.0, 0.0, 0.0])/100
mantle = np.array([44.3, 21.3, 2.3, 6.3, 2.5, 0.0, 0.0, 22.3, 0.0, 0.2, 0.0, 0.0, 0.0])/100
core = np.array([0.0, 0.0, 0.0, 84.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.6, 0.3, 0.6, 9.0])/100


######## DEFINE FUNCTIONS TO DO SOME DATA ANALYSIS OF SPREAD

# returns: nothing, but saves aggregated spread characteristics in output files: 
#           XYZ locations, radii and arrival times
# input: constituent parameters in the "data" array: [mass, constituent-nucleon cross-section, entry angle, # of particles to simulate]
def return_spread(data):
    mx = data[0]
    sigmand = data[1]

    print("== Run for mx = {} GeV, sigma = {} cm^2.".format(mx, sigmand))
    filesX = np.sort([str(f) for f in pathlib.Path().glob("Data/Xs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesY = np.sort([str(f) for f in pathlib.Path().glob("Data/Ys_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesZ = np.sort([str(f) for f in pathlib.Path().glob("Data/Zs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesV = np.sort([str(f) for f in pathlib.Path().glob("Data/Vs_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])
    filesT = np.sort([str(f) for f in pathlib.Path().glob("Data/Ts_mx-"+ str(mx) + "_sigma-"+ str(sigmand) +"_**")])

    ind = filesX[0].find("angle-")+len("angle-")
    angles = [f[ind:ind+15] for f in filesX]

    summary_array = []
    if len(filesX) == 0:
        return 0


    for i in range(len(filesX)):

        xst = np.load(filesX[i], allow_pickle=True)
        yst = np.load(filesY[i], allow_pickle=True)
        zst = np.load(filesZ[i], allow_pickle=True)
        vst = np.load(filesV[i], allow_pickle=True)
        ts = np.load(filesT[i], allow_pickle=True)

        # take the final locations
        xs = xst[-1]
        ys = yst[-1]
        zs = zst[-1]
        vs = vst[-1]

        rs = np.sqrt(xs**2 + ys**2 + zs**2)

        # print(ts)

        vhatfs = np.stack(np.array([xst[-1] - xst[-5], yst[-1] - yst[-5], zst[-1] - zst[-5]]), axis=1)

        # rescale things so everything is exactly at earth surface, but only if
        # constituents are less than a mean-free-path distance away from the surface 

        finalposns = np.array([go_to_boundary_new([xst[-1][i], yst[-1][i], zst[-1][i]], vhatfs[i], Rearth) for i in range(len(vhatfs))])
        xs = finalposns[:,0]
        ys = finalposns[:,1]
        zs = finalposns[:,2]

        rs_new = np.sqrt(xs**2 + ys**2 + zs**2)

        dist = rs_new - rs

        mfps = [lambdaMFP(rs[i], mx, sigmand, vs[i]) for i in range(len(rs))]

        ts_new = np.where(dist > mfps, np.nan, ts + dist/vs)
        ts = ts_new

        # print(ts)

        # here, calculate spread radius 

        avg_v = np.mean(vs)

        refx = np.mean(xs)
        refz = np.mean(zs)

        refy = np.sqrt(Rearth**2 - refx**2 - refz**2)*np.sign(sp.stats.mode(ys)[0])


        angles_x = np.array(np.arccos((xs*refx+ys*refy)/(np.linalg.norm(np.stack([xs, ys], axis = 1), axis=1)*np.linalg.norm([refx, refy]))))



        angles_z = np.array(np.arccos((ys*refy+zs*refz)/(np.linalg.norm(np.stack([ys, zs], axis = 1), axis=1)*np.linalg.norm([refy, refz]))))

        angles_z = np.where(np.isnan(angles_z), 1e-15, angles_z)
        angles_x = np.where(np.isnan(angles_x), 1e-15, angles_x)

        z_arc_dist = Rearth*angles_z*np.sign(zs)
        x_arc_dist = Rearth*angles_x*np.sign(xs)

        avg_x = np.mean(x_arc_dist)
        avg_z = np.mean(z_arc_dist)


        spread_rs = np.sqrt((x_arc_dist - avg_x)**2 + (z_arc_dist - avg_z)**2)

        z_score = np.abs(sp.stats.zscore(spread_rs, nan_policy='propagate'))

        z_score = np.where(np.isnan(z_score), 0, z_score)

        nonoutlier_indices = np.where(z_score < 5)[0]

        try:
            center = np.min(np.where(spread_rs[nonoutlier_indices] == np.min(spread_rs[nonoutlier_indices])))

        except ValueError:
            center = 0
        
        tcenter = ts[nonoutlier_indices][center]

        tdiff = [np.abs(t - tcenter) for t in ts[nonoutlier_indices]]

        loc1, alpha1 = list(sp.stats.maxwell.fit(spread_rs[nonoutlier_indices], floc=0))    

        max_r = max(np.max(spread_rs)*4, 0.001)
        rrange = np.linspace(0, max_r, 1000)
        maxwell_cdf = sp.stats.maxwell.cdf(rrange, loc=loc1, scale=alpha1)

        spreadsize = np.min(rrange[np.where(maxwell_cdf > 0.9)])

        if len(xst) < 3:
            spreadsize = np.nan

        summary_array.append([mx, sigmand, angles[i], spreadsize, avg_v])

        # save final XYZ, R, T, for each angle

        pd.to_pickle(np.array(xs[nonoutlier_indices]), "DataProcessed/FinalXs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(ys[nonoutlier_indices]), "DataProcessed/FinalYs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(zs[nonoutlier_indices]), "DataProcessed/FinalZs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(vs[nonoutlier_indices]), "DataProcessed/FinalVs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(spread_rs[nonoutlier_indices]), "DataProcessed/SpreadRs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))
        pd.to_pickle(np.array(ts[nonoutlier_indices]), "DataProcessed/FinalTs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, angles[i]))

    # save summary data, saved for all entry angles
    pd.to_pickle(np.array(summary_array), "DataProcessed/SummaryData_mx-{}_sigma-{}.pkl".format(mx, sigmand))

    print("All done saving!")

    return 0


######## RUNNING & PARALLELIZING THE CODE

cpus = int(multiprocessing.cpu_count())
print("Working across {} CPUs.".format(cpus))
# mxs = np.append(10**np.linspace(2, 10, 20), [10**np.float64(10.421052631578947), 10**np.float64(10.842105263157894), 10**np.float64(11.263157894736842), 10**np.float64(11.68421052631579), 10**np.float64(12.105263157894736)])
# # mxs = [1832.9807108324355]#[10**np.linspace(2, 10, 20)[2]]
# sigmas = 10**np.linspace(-41, -37, 20)
# mxs = [1832.9807108324355]#[10**np.linspace(2, 10, 20)[2]]
sigmas = [10**np.linspace(-41, -37, 20)[-5]]
# return_spread([mxs[0], sigmas[5]])
mxs = [1e3]
# sigmas = [1e-37, 1e-38]
args = []

for i in range(len(mxs)):
    for j in range(len(sigmas)):
        args.append([mxs[i], sigmas[j]])

start_time = time.time()

if __name__ == "__main__":
    
    # SET UP MULTIPROCESSING

    num_pool = cpus

    pool = multiprocessing.Pool(processes = num_pool, maxtasksperchild=2)
    
    doing_tasks = pool.map(return_spread, args)
    
    print("Processes are all done, in {} minutes!".format((time.time() - start_time)/60))
