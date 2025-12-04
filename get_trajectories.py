# %%
import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing
from scipy.integrate import quad
import os

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

######## DEFINE HELPFUL FUNCTIONS: SCATTERING IN THE EARTH

# returns: current layer in Earth
# input: radius from the centre of the Earth (km)
def layername(r):
    if r <= 1:
        return "centre"
    if r <= 3480*km:
        return "core"
    elif r <= 6346*km:
        return  "mantle"
    elif r < 6371*km:
        return "crust"
    else:
        return np.zeros_like(crust)

# returns: elemental composition of current layer in Earth
# input: radius from centre of the Earth (km)
def composition(r):
    if r <= 3480*km:
        return core
    elif r <= 6346*km:
        return  mantle
    elif r <= 6371*km:
        return crust
    else:
        return np.zeros_like(crust)

# returns: density (g/cm^3) of Earth material
# input: radius from centre of the Earth (km)
def rho(r):
    # renormalized radius used in PREM density equation
    x = r / (6371*km) 

    if r <= 1221.5*km:
        return 13.0885 - 8.8381*x**2
    elif r <= 3480*km:
        return 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
    elif r <= 5701*km:
        return 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3
    elif r <= 5771*km:
        return 5.3197 - 1.4836*x
    elif r <= 5971*km:
        return 11.2494 -8.0298*x
    elif r <= 6151*km:
        return 7.1089 - 3.8045*x
    elif r <= 6346.6*km:
        return 2.6910 + 0.6924*x
    elif r <= 6356*km:
        return 2.9
    elif r <= 6368*km:
        return 2.6
    elif r <= 6371*km:
        return 1.02
    else:
        return 0

# returns: number density of Earth material (atom/cm^3)
# input: radius from centre of the Earth (km)
def n_composition(r):
    ns = [(rho(r)*composition(r)[val]*gram)/(elements[val]*mp) for val in range(len(elements))]
    return ns

# returns: number density of a particular atom in Earth material (atom/cm^3)
# input: index of element as defined in "elements" array above; radius from centre of the Earth (km)
def nA(Aind, r):
    return n_composition(r)[Aind]

# returns: reduced mass of two masses a, b
# input: two masses
def mu(a, b):
    return a*b/(a+b)

# returns: nuclear radius in natural units, used in Helm form factor
# input: nuclear mass number A
def Rnuc(A):
    sk = 0.9/0.197 #(* nuclear skin thickness, GeV^-1 *)
    a =  0.52/0.197 #(* GeV^-1 *)
    R0 = (1.23*A**(1/3) - 0.6)/0.197 # GeV^-1
    Rnuc = (np.sqrt(R0**2 + (7/3)* np.pi**2* a**2 - 5*sk**2))

    return Rnuc

# returns: Helm form factor, squared 
# input: momentum transfer q, radius r, and nuclear skin thickness s
def helm2(q, r, s): #(* Helm form factor, squared, for momentum transfer q, radius r, and nuclear skin thickness s *)
    def j1(x):
        return np.sin(x)/x**2 - np.cos(x)/x; # spherical Bessel function
    Fa2 = (3*(j1(q*r)/(q*r)))**2*np.exp(-(q*s)**2)*np.heaviside(q*r-0.0001, 1)+np.heaviside(0.0001 - q*r, 1)

    return  Fa2

# returns: DM constituent-nucleus cross-section with some nucleus in the Earth, cm^2
# input: index of target nucleus, DM mass in GeV,  DM constituent-nucleon cross-section in cm^2, and DM velocity in km/s.
def sigmaAd(Aind, mx, sigmand, vel): # cm^2
    A = elements[Aind]
    mA = A*mp

    vel = vel/c

    Ermax = 2*mu(mx, mA)**2*(vel)**2/mA

    func_dsigmaAd = lambda Er: A**2 * sigmand * (mu(mA, mx)/mu(mp, mx))**2 * helm2(np.sqrt(2*mA*Er), Rnuc(A), s=0.9/0.197)*np.heaviside(np.sqrt(2)*vel*mu(mx, mA)- np.sqrt(2*mA*Er), 1)

    sigmaAd = mA/(2*(mu(mA, mx)*(vel))**2)*quad(func_dsigmaAd, 0, Ermax, epsabs=1.e-20, epsrel=1.e-20, limit=50)[0]
    return sigmaAd

# returns: local mean free path of the DM constituent, in km
# input: radius from centre of Earth in km, DM mass in GeV,  DM constituent-nucleon cross-section in cm^2, and DM velocity in km/s.
def lambdaMFP(r, mx, sigmand, vel):
    lambdaT = 0
    for Aind in range(len(elements)):
        lambdainv = nA(Aind, r)*sigmaAd(Aind, mx, sigmand, vel) # cm
        if lambdainv != 0:
            lambdaT += lambdainv
    return 1/lambdaT * cm

# returns: a number of randomly-chosen target atoms to use at the point of scattering, based on elemental abundances
# input: radius from centre of Earth in km, a number of desired samples
def get_target(r, num): # returns index of target atom
    targetcomp = composition(r)
    Aind = np.random.choice(range(len(elements)), p = targetcomp/np.sum(targetcomp), size=num)
    return Aind

# returns: sampled distance until next scatter for a set of constituents, in km
# input: constituent parameters (mx, sigmand), their radii from Earth centre (km) and velocities, 
#       the number of constituents to calculate this for
def sample_path_length(mx, sigmand, rs, vs, num_particles):

    mfps = np.array([lambdaMFP(rs[i], mx, sigmand, vs[i]) for i in range(len(rs))])
    # inverse sampling
    zeta = np.random.uniform(0, 1, num_particles)
    Ls = -np.log(1-zeta)*mfps

    return Ls

# returns: distance to the next boundary between Earth layers, along current trajectory
# input: current position, direction vector, radius of the boundary from Earth centre.
def dist_to_boundary_new(posn, vhat, Rboundary):
    # write equation of sphere in at^2 + bt + c form, with radius Rboundary
    # determine where this sphere intersects with the projected path along direction vector

    a = vhat[0]**2 + vhat[1]**2 + vhat[2]**2
    b = 2*vhat[0]*posn[0] + 2*vhat[1]*posn[1]+ 2*vhat[2]*posn[2]
    c = posn[0]**2 + posn[1]**2 + posn[2]**2 - Rboundary**2

    disc = b**2 - 4*a*c

    if disc < 0:
        return 0
    else:
        ts = np.roots([a, b, c])

        try:
            tval = np.min(ts[np.where(ts>=0)])
        except ValueError:
            return 0
        dx = vhat[0]*tval
        dy = vhat[1]*tval
        dz= vhat[2]*tval
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        return dist
    
######## DEFINE FUNCTIONS TO PERFORM SIMULATION OF TRAJECTORIES THROUGH EARTH

# returns: constituent trajectories through one single layer of the Earth:
#           their tracked positions and velocities over time, and the time taken to exit layer.
# input: constituent parameters (mx, sigmand), number of constituents
#           their initial positions, velocities upon entering layer,
#           radius of layer from centre of Earth, and name of layer ("crust", "core", or "mantle")
def get_trajectories_layer(mx, sigmand, num_particles, posns, vhats, vi, Rboundary, layer):

    xs = [posns[0]]
    ys = [posns[1]]
    zs = [posns[2]]
    vs = [vi]
    ttot = [np.zeros(num_particles)]
    scattered = [np.zeros(num_particles)]
    rs = np.sqrt(xs[-1]**2 + ys[-1]**2 + zs[-1]**2)

    vhatxs = vhats[0]
    vhatys = vhats[1]
    vhatzs = vhats[2]


    layers = np.array([layer for r in rs])

    dist_boundary = np.array([dist_to_boundary_new([xs[-1][i], ys[-1][i], zs[-1][i]], [vhatxs[i], vhatys[i], vhatzs[i]], Rboundary) for i in range(num_particles)])

    i = 0

    # we simulate particles until at least 99% of them are within 1 km of the layer's boundary. 
    while len(np.where(dist_boundary > 1)[0]) > 0.01*num_particles:


        active_particles = np.where(layers == layer)[0]


        Ls = sample_path_length(mx, sigmand, rs[active_particles], vs[-1][active_particles], len(active_particles))


        dist_boundary = np.array([dist_to_boundary_new([xs[-1][active_particles][i], ys[-1][active_particles][i], zs[-1][active_particles][i]], [vhatxs[active_particles][i], vhatys[active_particles][i], vhatzs[active_particles][i]], Rboundary) for i in range(len(active_particles))])


        scattered_particles = np.where(Ls < dist_boundary)[0]


        unscattered_particles = np.where(Ls >= dist_boundary)[0]
        scatters = len(scattered_particles)
        if scatters == 0:
            xdisp = np.zeros(num_particles)     
            ydisp = np.zeros(num_particles)     
            zdisp = np.zeros(num_particles)     
            xdisp[active_particles] = dist_boundary*vhatxs[active_particles]
            ydisp[active_particles] = dist_boundary*vhatys[active_particles]
            zdisp[active_particles] = dist_boundary*vhatzs[active_particles]
            rdisp = np.sqrt(xdisp**2 + ydisp**2 + zdisp**2)

            tdisp = rdisp/vs[-1]
            xs.append(xs[-1]+xdisp)
            ys.append(ys[-1]+ydisp)
            zs.append(zs[-1]+zdisp)
            rs = np.sqrt(xs[-1]**2 + ys[-1]**2 + zs[-1]**2)
            ttot.append(ttot[-1]+tdisp)
            scattered.append(scattered_particles)
        else:
            theta_cm = np.random.uniform(-np.pi, np.pi, size=scatters)
            phi = np.random.uniform(0, np.pi, size=scatters)
            targets = np.array([get_target(rs[active_particles[scattered_particles]][i], 1)[0] for i in range(scatters)])
            mAs = elements[targets]*mp
            theta_lab = np.arctan((mAs*np.sin(theta_cm))/(mx + mAs*np.cos(theta_cm)))
            xdisp = 0
            ydisp = 0
            zdisp = 0

            dist = Ls[scattered_particles]       

            xdisp = np.zeros(num_particles)     
            ydisp = np.zeros(num_particles)     
            zdisp = np.zeros(num_particles)     
            xdisp[active_particles[scattered_particles]] = dist*vhatxs[active_particles][scattered_particles]
            ydisp[active_particles[scattered_particles]] = dist*vhatys[active_particles[scattered_particles]]
            zdisp[active_particles[scattered_particles]] = dist*vhatzs[active_particles[scattered_particles]]
            xdisp[active_particles[unscattered_particles]] = dist_boundary[unscattered_particles]*vhatxs[active_particles[unscattered_particles]]
            ydisp[active_particles[unscattered_particles]] = dist_boundary[unscattered_particles]*vhatys[active_particles[unscattered_particles]]
            zdisp[active_particles[unscattered_particles]] = dist_boundary[unscattered_particles]*vhatzs[active_particles[unscattered_particles]]


            rdisp = np.sqrt(xdisp**2 + ydisp**2 + zdisp**2)

            tdisp = rdisp/vs[-1]


            xs.append(xs[-1]+xdisp)
            ys.append(ys[-1]+ydisp)
            zs.append(zs[-1]+zdisp)
            rs = np.sqrt(xs[-1]**2 + ys[-1]**2 + zs[-1]**2)

            ttot.append(ttot[-1]+tdisp)

            vs_new = np.copy(vs[-1])
            vs_new[scattered_particles] *= np.sqrt(1 - 2*(mx*mAs)/(mx+mAs)**2 * (1 - np.cos(theta_cm)))

            vs.append(vs_new)

            vhat_rs = np.sqrt(np.square(vhatxs) + np.square(vhatys) + np.square(vhatzs))

            vhat_theta = np.arccos(vhatzs/vhat_rs)
            vhat_phis = np.zeros_like(vhat_rs)

            xdivy = np.where(vhatxs==0, 0, vhatys/vhatxs)

            vhat_phis = np.where(vhatxs > 0, np.arctan(xdivy), vhat_phis)
            vhat_phis = np.where((vhatxs < 0) & (vhatys >= 0), np.arctan(xdivy) + np.pi, vhat_phis)
            vhat_phis = np.where((vhatxs < 0) & (vhatys < 0), np.arctan(xdivy) - np.pi, vhat_phis)
            vhat_phis = np.where(vhatxs == 0, np.pi/2, vhat_phis)

            vhatxs_new = np.copy(vhatxs)
            vhatys_new = np.copy(vhatys)
            vhatzs_new = np.copy(vhatzs)

            # get new velocity direction
            vhatxs_new[active_particles[scattered_particles]] = np.sin(theta_lab)*np.cos(phi)
            vhatys_new[active_particles[scattered_particles]] = np.sin(theta_lab)*np.sin(phi)
            vhatzs_new[active_particles[scattered_particles]] = np.cos(theta_lab)*np.ones_like(phi)

            # # # rotate it to be in the normal frame rather than in scatter frame
            vxn = np.copy(vhatxs_new[active_particles[scattered_particles]])
            vyn = np.copy(vhatys_new[active_particles[scattered_particles]])
            vzn = np.copy(vhatzs_new[active_particles[scattered_particles]])

            vxn = vhatxs_new[active_particles[scattered_particles]]*np.cos(vhat_theta[active_particles[scattered_particles]]) + vhatzs_new[active_particles[scattered_particles]]*np.sin(vhat_theta[active_particles[scattered_particles]])
            vzn = -1*vhatxs_new[active_particles[scattered_particles]]*np.sin(vhat_theta[active_particles[scattered_particles]]) + vhatzs_new[active_particles[scattered_particles]]*np.cos(vhat_theta[active_particles[scattered_particles]])

            vxn2 = np.copy(vxn)
            vyn2 = np.copy(vyn)
            vzn2 = np.copy(vzn)


            vxn2 = vxn*np.cos(vhat_phis[active_particles[scattered_particles]]) - vyn*np.sin(vhat_phis[active_particles[scattered_particles]])
            vyn2 = vxn*np.sin(vhat_phis[active_particles[scattered_particles]]) + vyn*np.cos(vhat_phis[active_particles[scattered_particles]])


            vhatxs_new[active_particles[scattered_particles]] = vxn2
            vhatys_new[active_particles[scattered_particles]] = vyn2
            vhatzs_new[active_particles[scattered_particles]] = vzn2

            vhatxs = vhatxs_new
            vhatys = vhatys_new
            vhatzs = vhatzs_new
        layers = np.array([layername(r) for r in rs])

        i += 1
    
    return xs, ys, zs, vhatxs, vhatys, vhatzs, vs, ttot

# returns: constituent trajectories through the whole Earth:
#           their tracked positions and velocities over time, and the time taken to exit the Earth.
# input: constituent parameters in the "data" array: [mass, constituent-nucleon cross-section, entry angle, # of particles to simulate]
def earth_trajectory(data):

    mx = data[0]
    sigmand = data[1]
    theta_entry = data[2]
    num_particles = data[3]
    xi = np.zeros(num_particles)
    yi = np.ones(num_particles)*(Rearth-1)
    zi = np.zeros(num_particles)
    vi = np.ones(num_particles)*300

    phi = np.pi/2
    theta_azim = np.pi/2

    vhatx = np.sin(-1*(np.pi - theta_azim - theta_entry))*np.cos(phi)*np.ones(num_particles)
    vhaty = np.sin(-1*(np.pi - theta_azim - theta_entry))*np.sin(phi)*np.ones(num_particles)
    vhatz = np.cos(-1*(np.pi - theta_azim - theta_entry))*np.ones(num_particles)

    print("== Run for mx = {} GeV, sigma = {} cm^2, theta = {} degrees.".format(mx, sigmand, theta_entry))

    # below, we simulate constituents first passing through the Earth crust, then mantle, then core, then mantle, then crust, until re-exit

    xs, ys, zs, vhatxs, vhatys, vhatzs, vs, ttot = get_trajectories_layer(mx, sigmand, num_particles, [xi, yi, zi], [vhatx, vhaty, vhatz], vi, Rmantle, "crust")


    xs2, ys2, zs2, vhatxs, vhatys, vhatzs, vs2, ttot2 = get_trajectories_layer(mx, sigmand, num_particles, [xs[-1], ys[-1], zs[-1]], [vhatxs, vhatys, vhatzs], vs[-1], Rcore , "mantle")

    xs += xs2
    ys += ys2
    zs += zs2

    xs3, ys3, zs3, vhatxs, vhatys, vhatzs, vs3, ttot3 = get_trajectories_layer(mx, sigmand, num_particles, [xs2[-1], ys2[-1], zs2[-1]], [vhatxs, vhatys, vhatzs], vs2[-1], Rcore+1, "core")
    xs += xs3
    ys += ys3
    zs += zs3

    xs4, ys4, zs4, vhatxs, vhatys, vhatzs, vs4, ttot4 = get_trajectories_layer(mx, sigmand, num_particles, [xs3[-1], ys3[-1], zs3[-1]], [vhatxs, vhatys, vhatzs], vs3[-1], Rmantle+1, "mantle")
    xs += xs4
    ys += ys4
    zs += zs4

    xs5, ys5, zs5, vhatxs, vhatys, vhatzs, vs5, ttot5 = get_trajectories_layer(mx, sigmand, num_particles, [xs4[-1], ys4[-1], zs4[-1]], [vhatxs, vhatys, vhatzs], vs4[-1], Rearth-1, "crust")
    xs += xs5
    ys += ys5
    zs += zs5

    tf = np.array(ttot[-1]) + np.array(ttot2[-1])+ np.array(ttot3[-1])+ np.array(ttot4[-1]) + np.array(ttot5[-1])

    # we save the data to pkl format. 
    pd.to_pickle(np.array(xs), "DataOct/Xs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, theta_entry))
    pd.to_pickle(np.array(ys), "DataOct/Ys_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, theta_entry))
    pd.to_pickle(np.array(zs), "DataOct/Zs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, theta_entry))
    pd.to_pickle(np.array(vs5), "DataOct/Vs_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, theta_entry))
    pd.to_pickle(np.array(tf), "DataOct/Ts_mx-{}_sigma-{}_angle-{}.pkl".format(mx, sigmand, theta_entry))

    print("Done saving!")

    return 0

######## RUNNING & PARALLELIZING THE CODE

# DEFINING RunParameters
cpus = int(os.environ['SLURM_CPUS_PER_TASK']) # set to however many CPUs you wish to parallelize over
print("Working across {} CPUs.".format(cpus))

# define the constituent mass and cross-section ranges to simulate
mxs_initial = np.append(10**np.linspace(2, 10, 20), [1e3, 2e3])
mxs_extended = 10**np.array([np.float64(10.421052631578947), np.float64(10.842105263157894), np.float64(11.263157894736842), np.float64(11.68421052631579), np.float64(12.105263157894736)])
mxs_specific = [1e6]

mxs = np.append(mxs_initial, mxs_extended)

sigmas = 10**np.linspace(-41, -37, 20)

# define the entry angles to simulate
# either sample from thetavals using the 2*sintheta*costheta distribution, or using particular angles 
thetavals = np.linspace(0, np.pi/2, 50)
thetaprobs = np.array([2*np.sin(th)*np.cos(th) for th in thetavals])

# here we show the particular angles used in this paper: we sampled them once, then kept them fixed for ease of plotting & comparison later.
thetaentries = [0.1282282715750, 0.2564565431501, 0.5129130863003, 0.5449701541941, 0.7052554936630, 0.7373125615567, 0.8014266973443, 0.8334837652381, 0.9296549689194, 0.9617120368132, 1.1219973762820, 1.1861115120696, 1.2181685799633, 1.2502256478571, 1.3784539194322]#np.random.choice(thetavals, p = thetaprobs/np.sum(thetaprobs), size=20)

# number of DM constituents to simulate
num_particles = 1000

# set up multi-processing arguments
args = []

for i in range(len(mxs)):
    for j in range(len(sigmas)):
        for k in range(len(thetaentries)):
            args.append((mxs[i], sigmas[j], thetaentries[k], num_particles))

# initialize positions of constituents when they first cross the Earth
xi = np.zeros(num_particles)*0
yi = np.ones(num_particles)*(Rearth-1)
zi = np.zeros(num_particles)*0

start_time = time.time()

if __name__ == "__main__":
    
    # MULTIPROCESSING

    num_pool = cpus

    pool = multiprocessing.Pool(processes = num_pool, maxtasksperchild=2)
    
    doing_tasks = pool.map(earth_trajectory, args)
    
    print("Processes are all done, in {} minutes!".format((time.time() - start_time)/60))
