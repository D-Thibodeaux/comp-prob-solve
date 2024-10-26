import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

def psi_2p_z(x, y, z):
    """
    calculate the value of psi for the 2pz orbital given values of x, y, and z.

    parameters: 
    x (float) : x coordinate in atomic units
    y (float) : y coordinate in atomic units
    z (float) : z coordinate in atomic units

    output:
    value of psi
    """
    #define value of bohr radius 
    a0 = 1
    #calculate constant = 1/(4 * sqrt(2pi) * a0)
    constant = 1 / (4 * np.sqrt(2 * np.pi) * a0)

    #calculate exponetial term = exp(-r/2a0) after converting r into cartesian coordinates
    exponential = np.exp(-np.sqrt(x**2 + y**2 + z**2) / (2 * a0))

    #finish calculating psi for 2pz by multipling constant, exponential and (r/a0) * cos(theta) converted to cartesian coordinates
    return constant * z/a0 * exponential

def random_sampling(L = 20, n_samples = 100, R = 2.0):
    """
    calculate the overlap integral of two hydrogen 2pz orbitals 

    parameters: 
    L (float) : region over which to sample (-L to L in all three directions)
    n_samples : number of samples from in region of 0 to L
    R (float) : distance between the two hydrogen nuclei in amu

    outputs:
    value of the overlap integral
    """
    #generate samples, only the first quadrandt needs to be sampled since the integrand is symmetrical around the center of the two orbitals
    np.random.seed(42)
    x = (np.random.rand(n_samples) * L)
    y = (np.random.rand(n_samples) * L)
    z = (np.random.rand(n_samples) * L)

    #calculate integrand at each point
    Integrand = psi_2p_z(x, y, z + R/2) * psi_2p_z(x, y, z - R/2)
    #calculate volume of space sampled
    volume = (2 * L) ** 3

    #return S(R) 
    return volume * np.average(Integrand)  

def importance_sampling(n_samples = 100, R = 2.0):
    """
    calculate the overlap integral of two hydrogen 2pz orbitals 

    parameters: 
    n_samples : number of samples from in region of 0 to L
    R (float) : distance between the two hydrogen nuclei in amu

    outputs:
    value of the overlap integral
    """
    np.random.seed(42)
    #generate samples using 
    x = expon.rvs(size=n_samples, scale=1)
    y = expon.rvs(size=n_samples, scale=1)
    z = expon.rvs(size=n_samples, scale=1)
    
    #caluclate the integrand's numerator and denominator
    
    Integrand_term_1 = psi_2p_z(x, y, z + R/2) 
    Integrand_term_2 = psi_2p_z(x, y, z - R/2)
    Integrand = Integrand_term_1 * Integrand_term_2
    denominator = expon.pdf(x) * expon.pdf(y) * expon.pdf(z)
    
    #calculate integrand
    Integrand = Integrand/denominator
    return np.average(Integrand) * 8


