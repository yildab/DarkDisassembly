import numpy as np
import pandas as pd
import time
import multiprocessing
import pathlib
import scipy as sp
import itertools
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

def sigmaAd2(Aind, sigmand, mx): # cm^2
    A = elements[Aind]
    mA = A*mp
    sigmaAd = A**2 * sigmand * (mu(mA, mx)/mu(mp, mx))**2# * helm2(q, Rnuc, sk) DO THIS PART LATER

    return sigmaAd
 
def lambdaMFP2(r, sigmand, mx): # in km
    lambdaT = 0
    for Aind in range(len(elements)):
        lambdainv = nA(Aind, r)*sigmaAd2(Aind, sigmand, mx) # cm
        if lambdainv != 0:
            lambdaT += lambdainv
    return 1/lambdaT * cm


# returns: a number of randomly-chosen target atoms to use at the point of scattering
#               based on elemental abundances & cross-sections
# input: radius from centre of Earth in km, a number of desired samples
def get_target(r, num, mx, sigmand, vel): # returns index of target atom
    targetcomp = composition(r)
    targetsigmaAd = np.array([sigmaAd(Aind, mx, sigmand, vel) for Aind in range(len(elements))])
    probs = (targetcomp * targetsigmaAd)/np.sum(targetcomp * targetsigmaAd)
    Aind = np.random.choice(range(len(elements)), p = probs, size=num)
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

######## DEFINE HELPFUL FUNCTIONS: DATA CLEANUP

# returns: projected locations on next boundary between Earth layers, travelling along current trajectory
# input: current position, direction vector, radius of the boundary from Earth centre.
def go_to_boundary_new(posn, vhat, Rboundary):
    # write equation of sphere in at^2 + bt + c form, with radius Rboundary
    # determine where this sphere intersects with the projected path along direction vector

    a = vhat[0]**2 + vhat[1]**2 + vhat[2]**2
    b = 2*vhat[0]*posn[0] + 2*vhat[1]*posn[1]+ 2*vhat[2]*posn[2]
    c = posn[0]**2 + posn[1]**2 + posn[2]**2 - Rboundary**2

    disc = b**2 - 4*a*c

    if disc < 0:
        return posn
    else:
        ts = np.roots([a, b, c])
        try:
            tval = np.min(ts[np.where(ts>=0)])
        except ValueError:
            return posn
        dx = vhat[0]*tval
        dy = vhat[1]*tval
        dz= vhat[2]*tval
        return [posn[0] + dx, posn[1] + dy, posn[2] + dz]
