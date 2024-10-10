import numpy as np
import scipy 
import pandas as pd

def W_iso(vi,vf, n = 1, T = 300, n_grid_pts = 1000):
    """
    calculate work done during an isothermal expansion of an ideal gas from a specified initial volume t a specified final volume. 
    Uses R = 8.314 J/mol-k

    Parameters:
    vi - (float) : initial volume of system in m^3
    vf - (float) : final volume of system in m^3
    n - (float) : number of moles of gas in system
    T - (float) : Temperature of system in K
    n_grid_pts - (int) : number of volume values to use for integration

    output:
    work in units of J
    """
    R = 8.314
    #build grid of volume points and deermine the spacing 
    volumes = np.linspace(vi,vf,n_grid_pts)
    dv = volumes[1]-volumes[0]

    #calculate set of pressure values 
    pressures = n*R*T / volumes

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
        work = W_iso(vi,vf)
        works.append(work)
    
    #make csv with work vs final volume
    df = pd.DataFrame()
    df['V_f (m^3)'] = final_volumes
    df['Work (J)'] = works
    df.to_csv('isothermal_data.csv', index = False)
