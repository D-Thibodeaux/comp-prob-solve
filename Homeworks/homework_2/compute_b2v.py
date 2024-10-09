import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate
from optimize_argon_dimer import lennard_jones
from ase.units import kB

Na = 6.022 * 10 ** 23

def hard_sphere(r, sigma = 3.4):
    """
    given a separation distance r calculate potential energy using the hard sphere potential.

    parameters:
    r (float) : pair separation distance in angstroms
    sigma (float) : diameter of hard spahere in angstroms

    output:
    potential energy (infinte if < sigma 0 in >= 0)
    """
    return 1000 if r < sigma else 0

def square_well(r, sigma = 3.4, epsilon = 0.01, lmbda = 1.5):
    """
    given a separation distance r calculate potential energy using the square well potential.

    parameters:
    r (float) : pair separation distance in angstroms
    sigma (float) : particle diameter in angstroms
    epsilon (float) : well depth in eV
    lmbda (float) : range of well - unitless

    output:
    potential energy (infinte if < sigma 0 in >= 0)
    """
    if r < sigma: return 1000
    if r >= lmbda * sigma: return 0
    else : return -epsilon

def b2v_calc(T, potential):
    """
    calculate the second viral coefficient given a temperature and potential function

    parameters: 
    T (float) : temperature in Kelvins
    potential (function) : previously potential functiopn to use

    returns:
    second virial coeeficient 
    """
    #define grid to integrate across and determine dx
    rs = np.linspace(1/10000, 5 * 3.4, 1000)
    dx = rs[1]-rs[0]

    #calculate energy at each value of r and calculate set of exponents 
    fx = np.array([potential(x) for x in rs]) 
    fx = -1.0 * fx / (kB * T)

    #calculate values to be used in integration
    integral_comps = (np.exp(fx) - 1) * rs ** 2
    
    #approximate integral using trapezoid rule
    integral = scipy.integrate.trapezoid(integral_comps, dx = dx)

    b2v = -2 * np.pi * Na * integral
    return b2v
    

if __name__ == '__main__':
    #define set of temperatures ranging from 100 to 800K in 1 K intervals
    temps = np.array(range(100,801))

    #setup dataframe and set of lists to store data in
    df = pd.DataFrame()
    df['T'] = temps
    hrdsph = []
    sqrwll = []
    lnrdjns = []

    #calculate virial coefficients at different temperatures for each potential
    for T in temps:
        for potential, data in zip([hard_sphere, square_well, lennard_jones], [hrdsph, sqrwll, lnrdjns]):
            value = b2v_calc(T, potential)
            data.append(value)
            if T == 100:
                print(f"second virial coefficient for {str(potential).rsplit(' ')[1].replace('_', ' ')} potential at 100 K is {value:.4g}")

    #make csv with coefficients at different temperatures
    df['hard sphere'] = hrdsph
    df['square well'] = sqrwll
    df['lennard jones'] = lnrdjns
    df.to_csv('homework-2-2/viral_coeff.csv', index = False)

    #plot b2v vs temperature for each potential
    fig = plt.figure(figsize = (8,6))
    plt.plot(temps, hrdsph, label = 'Hard Sphere')
    plt.plot(temps, sqrwll, label = 'Square Well')
    plt.plot(temps, lnrdjns, label = 'lennard Jones')
    plt.axhline(0, color = 'black', linestyle = '--', alpha = 0.5, zorder = 0)

    #format plot and save to png file
    plt.xlabel("Temperature (K)")
    plt.ylabel('$B_{2v}$')
    plt.title('$B_{2v}$ as a function of temperature')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('homework-2-2/b2v_vs_temp.png', format = 'png')
