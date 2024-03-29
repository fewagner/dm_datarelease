import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import expon, norm, uniform, gamma
from scipy import stats
from itertools import product, chain
import numba as nb
from scipy.optimize import fsolve, bisect, minimize, basinhopping
import matplotlib as mpl
from tqdm.auto import tqdm, trange
import pdb
from math import gamma as ga
from numba import vectorize, float64, njit
import argparse
SQRT2PI = np.sqrt(2.0 * np.pi)

# ----------------------------------------
# Input handling
# ----------------------------------------

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, help="Path to the directory.")
ap.add_argument("-i", "--idx_dataset", type=int, default=0, help="Index of the dataset: 0, ..., 9.")
ap.add_argument("-w", "--which_limit", type=str, default='a', help="Which experiments: a, b, c, ab, bc, ac or abc.")
ap.add_argument("-n", "--num", type=int, default=30, help="Nmbr of the points used for one limit.")

args = vars(ap.parse_args())
num = args['num']

# ----------------------------------------
# Functions
# ----------------------------------------

# TODO distribution elements should be vectorized instead for loops
# normal distribution
@njit
def pdf_normal(x, loc=0., scale=1.):
    u = (x - loc) / scale
    return np.exp(-0.5 * u ** 2) / (SQRT2PI * scale)
    
# exponential distribution
@njit
def pdf_expon(x, scale=1.):
    if 0 < x:
        return np.exp(-x/scale)/scale
    return 0

# uniform distribution
@njit
def pdf_uniform(x, loc=0., scale=1.):
    if loc <= x <= loc + scale:
        return 1 / scale
    return 0

# gamma distribution
@njit
def pdf_gamma(x, a=1., scale=1.):
    u = x ** (a - 1) * np.exp( - x / scale)
    return u / ga(a) / scale ** a

def inbounds(value, low, up):
    # utility function for bounds checks
    if value < low:
        return False, 10**50 + 10**50*np.abs(value - low)
    elif value > up:
        return False, 10**50 + 10**50*np.abs(value - up)
    else:
        return True, 0
    
@njit
def density_alice_nb(pars, 
                  x, 
                  lamb,
                  efficiency
                  ):
    """
    The density function for experiment A.
    """
    
    density = np.zeros(len(x))

    for i,v in enumerate(x):
        # signal
        density[i] += pars[0] * pdf_expon(v, lamb)
        #flat
        density[i] += pars[1] * pdf_uniform(v, loc=0., scale=20.)
        #peak
        density[i] += pars[2] * pdf_normal(v, loc=6, scale=0.5)
        #rise
        density[i] += pars[3] * pdf_expon(v, scale=0.1)
    
    #efficiency
    density *= efficiency
    
    return density

@njit
def density_bob_nb(pars, 
                  x, 
                  lamb,
                  efficiency
                  ):
    """
    The density function for experiment B.
    """
    
    density = np.zeros(len(x))

    for i,v in enumerate(x):  
        # signal
        density[i] += pars[0] * pdf_expon(v, lamb)
        #flat
        density[i] += pars[1] * pdf_uniform(v, loc=0., scale=10.)
        #peak
        density[i] += pars[2] * pdf_normal(v, loc=6, scale=0.5)
        #rise
        density[i] += pars[3] * pdf_expon(v, scale=0.1)
    
    #efficiency
    density *= efficiency
    
    return density

@njit
def density_carol_nb(pars, 
                  x, 
                  lamb,
                  efficiency
                  ):
    """
    The density function for experiment C.
    """
    
    density = np.zeros(len(x))

    for i,v in enumerate(x):  
        # signal
        density[i] += pars[0] * pdf_expon(v, lamb)
        #gamma
        density[i] += pars[1] * 0.6153846153846154 * pdf_gamma(v, a=2.25, scale=66.67)
        #peak
        density[i] += pars[1] * 0.12307692307692308 * pdf_normal(v, loc=45., scale=2.)
        #peak
        density[i] += pars[1] * 0.1846153846153846 * pdf_normal(v, loc=75., scale=3.)
        #peak
        density[i] += pars[1] * 0.07692307692307693 * pdf_normal(v, loc=120., scale=5.)
        #peak
        density[i] += pars[2] * pdf_normal(v, loc=200., scale=10.)
        #peak
        density[i] += pars[3] * pdf_normal(v, loc=300., scale=20.)
    
    #efficiency
    density *= efficiency
    
    return density

def nll_combined(pars, datas, lamb, densities, grids, eff_grid, efficiencies, sig_fixed, bnds, npars, exposures):
    """
    The negative extended log likelihood.
    
    grid: 1D array, the grid over the combined ROI of all measurements for the integration
    eff_grid: list of 1D arrays, the efficiencies of the measurements, evaluated at the grid values, 
        set to zero outside the ROI of the measurement
    efficiencies: list of 1D arrays, the efficiencies of the measurements, evaluated at the x values, 
        set to zero outside the ROI of the measurement
    """
    
    n = len(npars)  # the number of densities/measurements
    nll = 0  # init the count variable for the likelihood
    nu = 0  # init the count variable for the integral over the density
    
    # loop over the measurements
    for i in range(n):
    
        # get the correct pars and bounds for this measurement
        pars_ = list([pars[0]]) + list(pars[int(np.sum(npars[:i]) - i + 1):int(np.sum(npars[:i+1]) - i)])
        bnds_ = list([bnds[0]]) + list(bnds[int(np.sum(npars[:i]) - i + 1):int(np.sum(npars[:i+1]) - i)])
        
        # if the signal weight is fixed, set it for all the densities
        if sig_fixed is not None:
            pars_[0] = sig_fixed  # fixing the parameter here is probably the issue! either use Miniuit (can fix parameters) or give DM not as par but as arg
            
        # pdb.set_trace()
    
        # check if the parameters are within their bounds
        for p, (low, up) in zip(pars_, bnds_):
            ok, retval = inbounds(p, low, up)
            if not ok:
                return retval
    
        # calculate and sum up the likelihood and the integral over the density
        
        lh = densities[i](pars_, datas[i], lamb, efficiencies[i])#/exposures[i]
        nll -= np.sum(np.log(lh))
        nu += np.trapz(y=densities[i](pars_, grids[i], lamb, eff_grid[i]),
                       x=grids[i])#/exposures[i]  # integrate in roi
        
    return nu + nll  # this is nu - sum(log(lh)), i.e. the negative extended log likelihood

def get_limit_lh(dm_pars, densities, x0s, datas, grids, efficiencies, 
                 exposures, bndss, ):
    
    # get the number of measurements
    npars = [len(x0) for x0 in x0s]
    
    # combine all start values, datas and bounds into a 1D array
    x0s = [0] + list(chain.from_iterable([x0[1:] for x0 in x0s]))
    bnds = [(0, 5e100)] + list(chain.from_iterable([b[1:] for b in bndss]))
    
    # def print_fun(x, f, accepted):
    #     print("at minimum %.4f accepted %d" % (f, int(accepted)))
    # def print_fun(xk):
    #     print(f"parameters: {xk}")
        
    # here we start with the actual limit calculation
    limit = []    
    for p in tqdm(dm_pars):
        
        # the nll_combined function takes 9 arguments additionally to the parameters
        args_ = (datas,  # x
                 p,  # lamb
                 densities,
                 grids,
                 efficiencies,  # eff_grid
                 [np.interp(d,g,e) for d,g,e in zip(datas,grids,efficiencies)],  # efficiency
                 None,  # sig_fixed
                 bnds,  # bnds
                 npars,
                 exposures,  # exposures
                 )
    
        # calculate best fit
        # TODO very important that this is really good!!
        # res_best = minimize(nll_combined, x0=x0s, args=args_, callback=print_fun, method='Nelder-Mead', options={'fatol': 0.01})
        res_best = basinhopping(nll_combined, x0=x0s, minimizer_kwargs={'args': args_}), #niter=3, stepsize=10, T=100, 
                                # callback=print_fun)
        
        print('Res best: ', res_best)
        print('Res best: ', res_best.x)

        # do exclusion fit
        def implfunc(val): 

            # the nll_combined function takes 9 arguments additionally to the parameters
            args_excl = (datas,  # x
                 p,  # lamb
                 densities,
                 grids,  # grids
                 efficiencies,  # eff_grid
                 [np.interp(d,g,e) for d,g,e in zip(datas,grids,efficiencies)],  # efficiency
                 val,# sig_fixed
                 bnds,  # bnds
                 npars,  # npars
                 exposures,  # exposures
                 )

            # TODO here we should maybe use the best fit pars as x0!!
            # TODO activate flag adaptive
            res = minimize(nll_combined, x0=x0s, args=args_excl)#, callback=print_fun, method='Nelder-Mead', options={'fatol': 0.01})
            # res = basinhopping(nll_combined, x0=res_best.x, minimizer_kwargs={'args': args_excl}, #niter=3, stepsize=10, T=100, 
            #                    callback=print_fun)

            return res.fun - (res_best.fun + 1.282**2/2)  # results are negativ, this is llh_best - Z^2/2 - llh_excl l 
        
        # pdb.set_trace()
        
        # try:
        res_excl = bisect(implfunc, a=res_best.x[0], b=1e9, maxiter=1000)  # put b maybe to 2*best fit?? and only if this doesnt work make upper bound 10 times higher
        limit.append(res_excl)
        # except:
        #     limit.append(np.nan)
    
    return limit

# ----------------------------------------
# Constants
# ----------------------------------------

NAMES = ['A', 'B', 'C']
EXPOSURE = np.array([7, 5, 100])
COUNTS = np.array([2100, 1000, 2000])
THRESHOLD_LOW = np.array([0.1, 0.035, 1])
THRESHOLD_UP = np.array([20, 10, 400])
RESOLUTION = np.array([0.015, 0.005, 0.15])
EFFICIENCY = np.array([0.8, 0.65, 0.5])
NMBR_REPETITIONS = 10

data_alice = []
data_bob = []
data_carol = []

for i in range(NMBR_REPETITIONS):
    data_alice.append(np.loadtxt(args['path'] + 'data/alice/data_alice_{}.txt'.format(i)))
    data_bob.append(np.loadtxt(args['path'] + 'data/bob/data_bob_{}.txt'.format(i)))
    data_carol.append(np.loadtxt(args['path'] + 'data/carol/data_carol_{}.txt'.format(i)))
    
grids = []
efficiencies = []

grids.append(np.loadtxt(args['path'] + 'data/alice/efficiency_alice.txt')[:,0])
grids.append(np.loadtxt(args['path'] + 'data/bob/efficiency_bob.txt')[:,0])
grids.append(np.loadtxt(args['path'] + 'data/carol/efficiency_carol.txt')[:,0])

efficiencies.append(np.loadtxt(args['path'] + 'data/alice/efficiency_alice.txt')[:,1])
efficiencies.append(np.loadtxt(args['path'] + 'data/bob/efficiency_bob.txt')[:,1])
efficiencies.append(np.loadtxt(args['path'] + 'data/carol/efficiency_carol.txt')[:,1])

data_alice = np.array(data_alice)
data_bob = np.array(data_bob)
data_carol = np.array(data_carol)
grids = np.array(grids)
efficiencies = np.array(efficiencies)
    
# ----------------------------------------
# parameters
# ----------------------------------------
    
bnds_alice = [(0, 5e100),  # w0   # bounds from zero to number of events
              (0, 5e100),  # w1
              (0, 5e100),  # w2
              (0, 5e100),  # w3
             ]

x0_alice = np.array([0.*len(data_alice[0])/EFFICIENCY[0],  # w0
            0.4*len(data_alice[0])/EFFICIENCY[0],  # w1
            0.3*len(data_alice[0])/EFFICIENCY[0],  # w2
            0.3*len(data_alice[0])/EFFICIENCY[0],  # w3 
                    ])

bnds_bob = [(0, 5e100),  # w0 
              (0, 5e100),  # w1
              (0, 5e100),  # w2
              (0, 5e100),  # w3
             ]

x0_bob = np.array([0.*len(data_bob[0])/EFFICIENCY[1],  # w0
            0.35*len(data_bob[0])/EFFICIENCY[1],  # w1
            0.3*len(data_bob[0])/EFFICIENCY[1],  # w2
            0.35*len(data_bob[0])/EFFICIENCY[1],  # w3 
                  ])

bnds_carol = [(0, 5e100),  # w0 
              (0, 5e100),  # w1
              (0, 5e100),  # w2
              (0, 5e100),  # w3
             ]

x0_carol = np.array([0.*len(data_carol[0])/EFFICIENCY[2],  # w0
            0.4*len(data_carol[0])/EFFICIENCY[2],  # w1
            0.15*len(data_carol[0])/EFFICIENCY[2],  # w2
            0.1*len(data_carol[0])/EFFICIENCY[2],  # w3 
                    ])

# ----------------------------------------
# Do the calculation
# ----------------------------------------

i = args['idx_dataset']

print('Calculate individual Limits...')
pars_alice = np.logspace(-1.5,1,num=num)  
pars_bob = np.logspace(-2,1,num=num)
pars_carol = np.logspace(-0.5,2,num=num)

if args['which_limit'] == 'a':

    limit_alice = get_limit_lh(dm_pars=pars_alice, 
                 densities=[density_alice_nb,], 
                 x0s=[x0_alice,], 
                 datas=[data_alice[i],],
                 grids=grids[[0,]], 
                 efficiencies=efficiencies[[0,]], 
                 exposures=EXPOSURE[[0,]], 
                 bndss=[bnds_alice,],
                              )/EXPOSURE[0]
    
    np.savetxt(args['path'] + f'data/limit/limit_a_{i}.txt', np.array([pars_alice,limit_alice, ]).T)

if args['which_limit'] == 'b':
    
    limit_bob = get_limit_lh(dm_pars=pars_bob, 
                 densities=[density_bob_nb,], 
                 x0s=[x0_bob,], 
                 datas=[data_bob[i],],
                 grids=grids[[1,]], 
                 efficiencies=efficiencies[[1,]], 
                 exposures=EXPOSURE[[1,]], 
                 bndss=[bnds_bob,],
                            )/EXPOSURE[1]

    np.savetxt(args['path'] + f'data/limit/limit_b_{i}.txt', np.array([pars_bob,limit_bob, ]).T)
    
if args['which_limit'] == 'c':
    
    limit_carol = get_limit_lh(dm_pars=pars_carol, 
                 densities=[density_carol_nb,], 
                 x0s=[x0_carol,], 
                 datas=[data_carol[i],],
                 grids=grids[[2,]], 
                 efficiencies=efficiencies[[2,]], 
                 exposures=EXPOSURE[[2,]], 
                 bndss=[bnds_carol,],
                              )/EXPOSURE[2]
    
    np.savetxt(args['path'] + f'data/limit/limit_c_{i}.txt', np.array([pars_carol,limit_carol, ]).T)

print('Calculate two-experiment Limits...')
pars_ab = np.logspace(-2,1,num=num)
pars_ac = np.logspace(-1.5,1.8,num=num)
pars_bc = np.logspace(-2,2,num=num)

if args['which_limit'] == 'ab':

    limit_ab = get_limit_lh(dm_pars=pars_ab, 
             densities=[density_alice_nb, density_bob_nb], 
             x0s=[x0_alice, x0_bob], 
             datas=[data_alice[i], data_bob[i]],
             grids=grids[[0,1]], 
             efficiencies=efficiencies[[0,1]], 
             exposures=EXPOSURE[[0,1]], 
             bndss=[bnds_alice, bnds_bob],
                        )/np.sum(EXPOSURE[[0,1]])
    
    np.savetxt(args['path'] + f'data/limit/limit_ab_{i}.txt', np.array([pars_ab,limit_ab, ]).T)

if args['which_limit'] == 'ac':
    
    limit_ac = get_limit_lh(dm_pars=pars_ac, 
             densities=[density_alice_nb, density_carol_nb], 
             x0s=[x0_alice, x0_carol], 
             datas=[data_alice[i], data_carol[i]],
             grids=grids[[0,2]], 
             efficiencies=efficiencies[[0,2]], 
             exposures=EXPOSURE[[0,2]], 
             bndss=[bnds_alice, bnds_carol],
                        )/np.sum(EXPOSURE[[0,2]])
    
    np.savetxt(args['path'] + f'data/limit/limit_ac_{i}.txt', np.array([pars_ac, limit_ac, ]).T)

if args['which_limit'] == 'bc':
    
    limit_bc = get_limit_lh(dm_pars=pars_bc, 
             densities=[density_bob_nb, density_carol_nb, ], 
             x0s=[x0_bob, x0_carol,], 
             datas=[data_bob[i], data_carol[i], ],
             grids=grids[[1,2]], 
             efficiencies=efficiencies[[1,2]], 
             exposures=EXPOSURE[[1,2]], 
             bndss=[bnds_bob, bnds_carol],
                        )/np.sum(EXPOSURE[[1,2]])
    
    np.savetxt(args['path'] + f'data/limit/limit_bc_{i}.txt', np.array([pars_bc, limit_bc, ]).T)

print('Calculate combined Limit ...')
pars_all = np.logspace(-2,2,num=num)

if args['which_limit'] == 'abc':

    limit_all = get_limit_lh(dm_pars=pars_all, 
             densities=[density_alice_nb, density_bob_nb, density_carol_nb, ], 
             x0s=[x0_alice, x0_bob, x0_carol,], 
             datas=[data_alice[i], data_bob[i], data_carol[i], ],
             grids=grids[[0,1,2]], 
             efficiencies=efficiencies[[0,1,2]], 
             exposures=EXPOSURE[[0,1,2]], 
             bndss=[bnds_alice, bnds_bob, bnds_carol],
                         )/np.sum(EXPOSURE)

    np.savetxt(args['path'] + f'data/limit/limit_all_{i}.txt', np.array([pars_all, limit_all, ]).T)