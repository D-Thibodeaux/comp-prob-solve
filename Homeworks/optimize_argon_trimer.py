import numpy as np
from scipy.optimize import minimize
from ase.io import write
from ase import Atoms
import matplotlib.pyplot as plt
from homework_1_2 import compute_bond_length, compute_bond_angle

#define a function to determine the the energy of the three Argon particle system
def lennard_jones_trimer (parameters, epsilon=0.01, sigma=3.4):
    x2, x3, y3 = parameters

    pos_1 = np.array([0,0,0])
    pos_2 = np.array([x2,0,0])
    pos_3 = np.array([x3,y3,0])

    r12 = compute_bond_length(pos_1, pos_2)
    r13 = compute_bond_length(pos_1, pos_3)
    r23 = compute_bond_length(pos_2, pos_3)

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

print("The bond distances are all approximately equal and all bond angles are 60 degrees. This means that the Argons are arranged in an equilateral triangle.")

#make extxyz file to show optimum structure
Ar = Atoms('Ar3', pos_set)
write("trimer.extxyz", Ar, format = 'extxyz')