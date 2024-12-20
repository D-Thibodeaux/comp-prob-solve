import numpy as np
from simulation import minimum_image

def calculate_end_to_end_distance(positions):
    """
    calculate end to end distance of the polymer chain

    parameters:
    positions (array) : array containing the positions of all the particles

    output:
    distance between the first and last particles in the chain
    """
    Ree = np.linalg.norm(positions[-1] - positions[0])
    return Ree

def calculate_radius_of_gyration(positions):
    """
    calculate the radius of gyration of the polymer chain

    parameters:
    positions (array) : array containing the positions of all the particles

    output:
    radius of gyration for the given position array
    """
    center_of_mass = np.mean(positions, axis=0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis=1))
    Rg = np.sqrt(Rg_squared)
    return Rg

def calculate_potential_energy(positions, box_size, k, r0, epsilon, sigma):
    """
    calculate potential energy using all three potentials (harmonic, attractive lennard-jones, and repulsive leenard-jones)

    parameters:
    positions (array) : array containing the positions of all the particles
    box_size (float) : side length of the cubic simulation box
    k (float) : spring constant
    r0 (float) : equilibrium distance between two particles
    epsilon (dictionary) : dictionary containing attractive and repulsive epsilon values
    sigma (float) : sigma vlaue for leard jones equation

    output:
    potential energy for the provided configuration
    """
    #calculate Harmonic bond potential energy:
    U_harmonic = 0
    for i in range(len(positions) - 1):
        #calculate displacement accounting for periodic boundaries
        displacement = positions[i+1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        #calculate distance
        distance = np.linalg.norm(displacement)
        #add to harmonic potential energy
        U_harmonic += 0.5 * k * (distance - r0) ** 2

    #calculate attractive and repulsive LJ energies\

    U_repulsive = 0
    U_attractive = 0

    #loop over all pairs without double counting
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            #calculate displacement accounting for periodic boundaries
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)

            #calculate repulsive energy
            if np.abs(i - j) == 2 and distance < 3 * sigma:
                eps = epsilon['repulsive']
                if distance >= 2.6 * sigma: 
                    U_repulsive += 0
                else:
                    U_repulsive += 4 * eps * ((sigma/distance) ** 12 - (sigma/distance) ** 6 + 0.25)

            #calculate attractive energy
            elif np.abs(i-j) > 2 and distance < 3 * sigma:
                eps = epsilon['attractive']
                U_attractive += 4 * eps * ((sigma/distance) ** 12 - (sigma/distance) ** 6)

            else:
                continue
    U_total = U_attractive + U_repulsive + U_harmonic

    return U_total
