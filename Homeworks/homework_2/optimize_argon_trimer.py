import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from homework_1_2 import compute_bond_length, compute_bond_angle

#define a function to determine the the energy of the three Argon particle system
def lennard_jones_trimer (parameters, epsilon=0.01, sigma=3.4):
    """
    Given a set of points, calculate the energy of an Argon trimer using a lennard jones potential.
    this function assumes the following:
    one particle (particle 1) is at the position [0,0,0]
    one particle (particle 2) is at [x2,0,0] where x2 is a variable that corresponds to the distance between particles 1 and 2
    one particle (particle 3) is at [x3,y3,0] where x3 and y3 correspond to any set of x and y positions

    Parameters:
    parameters (3 item array) : an array containg the values x2,x3 and y3 in that order. each of these values may be a float or an integer
    epsilon (float) : a value corresponding to the epsilon value in the lennard jones 12 6 potential. has units of eV
    sigma (float) :  a value corresponding to the sigma value in the lennard jones 12 6 potential. has units of angstroms

    returns: 
    potential energy of the system in eV
    """
    #unpack parameters
    x2, x3, y3 = parameters

    #defime positions
    pos_1 = np.array([0,0,0])
    pos_2 = np.array([x2,0,0])
    pos_3 = np.array([x3,y3,0])

    #compute distances from positions
    r12 = compute_bond_length(pos_1, pos_2)
    r13 = compute_bond_length(pos_1, pos_3)
    r23 = compute_bond_length(pos_2, pos_3)

    #calculate energ
    V_lj = 0
    for r in [r12, r13, r23]:
        V_lj += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    
    return V_lj

#find optimum coordinates
optimum_values = minimize(fun=lennard_jones_trimer, x0=(4,3,3), method="Nelder-Mead", tol=1e-6)['x']
print(optimum_values)

#build set of coordinates as numpy array
pos_set = np.array([[0, 0, 0], [optimum_values[0], 0, 0], [optimum_values[1], optimum_values[2], 0]])

#print out relevant information
print("\nR12 = ", np.sqrt((pos_set[1]-pos_set[0]).dot((pos_set[1]-pos_set[0]))))
print("R13 = ", np.sqrt((pos_set[2]-pos_set[0]).dot((pos_set[2]-pos_set[0]))))
print("R23 = ", np.sqrt((pos_set[2]-pos_set[1]).dot((pos_set[2]-pos_set[1]))), '\n')

print('θ123 = ', compute_bond_angle(pos_set[0], pos_set[1], pos_set[2]))
print('θ231 = ', compute_bond_angle(pos_set[1], pos_set[2], pos_set[0]))
print('θ312 = ', compute_bond_angle(pos_set[2], pos_set[0], pos_set[1]), '\n')

print("The bond distances are all equal and all bond angles are 60 degrees. This means that the Argons are arranged in an equilateral triangle.")

#make xyz file to show optimum structure
file = open('trimer.xyz', 'wt')
file.write(str(len(pos_set)) + "\n")
file.write("Argon trimer file\n")
for pos in pos_set:
    file.write(f"Ar   {pos[0]}   {pos[1]}   {pos[2]}\n")
