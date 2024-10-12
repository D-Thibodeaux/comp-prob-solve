import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt

k = 6.241509074461 * 10**18 * scipy.constants.k

def isolated_Ce(T):
    """
    compute partition function for an electron in the f orbital of an isolated Ce atom and calculate Internal energy, free energy and entropy

    parameters
    T (float) : temperature in Kelvin

    output:
    Internal energy (ev), free energy(ev) and entropy (ev/K)
    """
    #z is calculated from one state with an energy of 0 and degeneracy of 14 -> Z will always be 14
    Z = 14

    #U = dln(Z)/dB = 0 for a constant Z value
    U = 0

    #F = -kB T ln(Z)
    F = -k * T * np.log(Z)

    #S = -dF/dT, Z is constant with respect to Temperature, so: 
    S = k * np.log(Z)
    
    return (U,F,S)

def SOC_Ce(T):
    """
    compute partition function for an electron in the f orbital of a Ce atom with Spin-Orbit Coupling and calculates Internal energy, free energy and entropy

    parameters
    T (float) : temperature in Kelvin

    output:
    Internal energy (ev), free energy(ev) and entropy (ev/K)
    """
    beta = 1/(k*T)
    #z iz sum of 2F5/2 (E = 0) boltzmann factors (6 * e ^ 0) and 2F7/2 states giving:
    Z = 6 + 8 * np.exp(-(0.28)*beta)

    #U = dln(Z)/dB, where B = 1/kBT. by putting the expression for Z into an online derivative calculator, the following was obtained:
    U = 28 / (75 * np.exp((7 *beta) / 25) + 100)

    #F = kB T ln(Z)
    F = -k * T * np.log(Z)
    
    #S = -dF/dt. using an online derivative calculator and the previous expression for Z, the following was obtained:
    S = k * np.log(8 * np.exp(-7 / (25 * k * T)) + 6) + (56 * np.exp(-7 / (25 * k * T))) / (25 * T * (8 * np.exp(-7 / (25 * k * T)) + 6))

    return (U,F,S)

def CFS_SOC_Ce(T):
    """
    compute partition function for an electron in the f orbital of a Ce atom with Spin-Orbit Coupling and Crystal Field Splitting and calculates Internal energy, free energy and entropy

    parameters
    T (float) : temperature in Kelvin

    output:
    Internal energy (ev), free energy(ev) and entropy (ev/K)
    """
    beta = 1/(k*T)
    
    energies = np.array([0, 0, 0, 0, 0.12, 0.12, 0.25, 0.25, 0.32, 0.32, 0.32, 0.32, 0.46, 0.46])

    bf = np.exp(-energies * beta)
    Z = np.sum(bf)

    #U is also equal to <E> which is SUM(Ei * P_i) where P_i = exp(-E_i*beta)/Z
    U = np.sum(energies * bf) / Z

    #F = kB T ln(Z)
    F = -k * T * np.log(Z)

    #S can also be defined as -kB * SUM(P_i*ln(P_i)) where P_i = exp(-E_i*beta)/Z.
    S = -k * np.sum(bf/Z*np.log(bf/Z))

    return (U,F,S)

if __name__ == "__main__":
    #calculate thermodynamic values for each system for temperatures ranging from 300 to 2000
    Ts = np.linspace(300,2000, 1000)

    #set up plot with a subplot for each thermodynamic value
    fig, ax = plt.subplot_mosaic([[0],[1],[2]], figsize = (6, 18))

    #set up dataframe to store results
    df = pd.DataFrame()
    df['Temperature (K)'] = Ts

    for system in [isolated_Ce, CFS_SOC_Ce, SOC_Ce]:
        #get system name
        sys = str(system).rsplit(' ')[1]

        #calculate and store relevant values for each temperature
        Us = []
        Fs = []
        Ss = []
        for T in Ts:
            U,F,S = system(T)
            Us.append(U)
            Fs.append(F)
            Ss.append(S)

        #store calculated values in dataframe
        df[f"U-{sys} (eV)"] = Us
        df[f"F-{sys} (eV)"] = Fs
        df[f"S-{sys} (eV/K)"] = Ss

        #plot caluculated values
        ax[0].plot(Ts, Us, label = sys)
        ax[1].plot(Ts, Fs, label = sys)
        ax[2].plot(Ts, Ss, label = sys)

    #format and store plot
    ax[0].set_xlabel('Temperature (K)')
    ax[0].set_ylabel('Internal energy (eV)')
    ax[0].set_title('Internal Energy vs Temperature')
    ax[0].legend()

    ax[1].set_xlabel('Temperature (K)')
    ax[1].set_ylabel('Free energy (eV)')
    ax[1].set_title('Free Energy vs Temperature')
    ax[1].legend()

    ax[2].set_xlabel('Temperature (K)')
    ax[2].set_ylabel('Entropy (eV/K)')
    ax[2].set_title('Entropy vs Temperature')
    ax[2].legend()  

    plt.tight_layout()
    plt.savefig('Thermodynamic_Values.png', format = 'png')

    #Store dataframe as csv
    df.to_csv("Thermodynamic_data.csv", index = False)
