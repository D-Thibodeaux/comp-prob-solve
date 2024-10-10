import numpy as np
import scipy 
import pandas as pd

def W_adi(vi,vf, n = 1, T = 300, γ = 1.4, n_grid_pts = 1000):
    """
    calculate work done during an adiabatic expansion  of an ideal gas from a specified initial volume t a specified final volume. 
    Uses R = 8.314 J/mol-k

    Parameters:
    vi - (float) : initial volume of system in m^3
    vf - (float) : final volume of system in m^3
    n - (float) : number of moles of gas in system
    T - (float) : Temperature of system in K
    γ - (float) : adiabatic index
    n_grid_pts - (int) : number of volume values to use for integration

    output:
    work in units of J
    """
    R = 8.314
    volumes = np.linspace(vi,vf,n_grid_pts)
    dv = volumes[1]-volumes[0]

    #since PV**γ is constant, P_k = P_j * (V_j/V_k)**γ 
    #using this, calculate P values at each volume
    pressures = []
    #find initial pressure 
    Pi = n * R * T / vi
    pressures.append(Pi)

    for i, v in enumerate(volumes):
        #skip initial pressure
        if i == 0: continue
        #caluclate pressure
        P = pressures[i-1] * (volumes[i-1]/v) ** γ

        pressures.append(P)
    
    #calculate work using scipy trapezoidal integration
    work =  -1 * scipy.integrate.trapezoid(pressures, volumes, dv)
    return work

if __name__ == "__main__":
    #find the maximum value of vf 
    vi = 0.1
    vf_max = 3 * vi
    final_volumes = np.linspace(vi,vf_max, 1000)
    works = []
    #caluclate work due to expansion for values of vf ranging from vi to vf_max
    for vf in final_volumes:
        #compute work using defaults values
        work = W_adi(vi,vf)
        works.append(work)
    
    #make csv with work vs final volume
    df = pd.DataFrame()
    df['V_f (m^3)'] = final_volumes
    df['Work (J)'] = works
    df.to_csv('adiabatic_data.csv', index = False)