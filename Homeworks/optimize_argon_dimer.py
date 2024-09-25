import numpy as np
from scipy.optimize import minimize
from ase.io import write
from ase import Atoms
import matplotlib.pyplot as plt
from homework_1_2 import compute_bond_length, compute_bond_angle

#define Lennard Jones potential with preset epsilon and sigma values
def lennard_jones(r, epsilon=0.01, sigma=3.4):
    V_lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return V_lj

#calculate optimum pair separation distance
optimum_length = minimize(fun=lennard_jones, x0=4, method="Nelder-Mead", tol=1e-6)['x'][0]
print(optimum_length)

#build an extxyz file to view the optimum structure
pos = np.array([[0, 0, 0], [0, optimum_length, 0]])
Ar = Atoms('Ar2', pos)
write("dimer.extxyz", Ar, format = 'extxyz')

#build a set of radius values
r_values = np.linspace(3, 6, 300)

#plot Lennard Jones potential with a labelled line at the optimum separation distance
plt.plot(r_values , lennard_jones(r_values))
plt.axvline(optimum_length, linestyle = "--", color = 'red', label = f"{optimum_length:.2f}")

#format plot
plt.title("Potential Energy vs Ar-Ar distance")
plt.xlabel("radius (Å)")
plt.ylabel("Energy (eV)")
plt.legend()

#show plot
plt.tight_layout()
plt.show()