# Import necessary libraries
import numpy as np
from simulation import *
from analysis_functions import *
from ase import Atoms
from ase.io import write
import pandas as pd
import time

# Simulation parameters
dt = 0.01  # Time step
total_steps = 11000  # Number of steps
box_size = 100.0  # Size of the cubic box
k = 1.0 # Spring constant
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon = {
    'repulsive' : 1.0,  # Depth of repulsive LJ potential
    'attractive' : 0.5,  # Depth of attractive LJ potential
}
sigma = 1.0  # LJ potential parameter

#build array of temperatures to simulate
target_temperatures = np.linspace(0.1,1.0, 15)  # Target temperature
#loop over k values
for k in [0.5,1.0, 1.5,0.25]:
    #if not 1.25> k > 0.25: continue
    print(k)
    #loop over repulsive epsilon values
    for eps_repulsive in [1.0,1.5,2.0,2.5,3.0,4.0]:
        #if eps_repulsive < 4.0: continue
        #make arrays to store parameters
        Rgs = []
        Rees = []
        Us = []
        #if eps_repulsive <= 1.50: continue
        epsilon['repulsive'] = eps_repulsive
        print(epsilon)
        for target_temperature in target_temperatures:
            Rg = []
            Re = []
            U = []
            start = time.time()
            trajs = []
            # Initialize positions and velocities
            positions = initialize_chain(n_particles, box_size, r0)
            velocities = initialize_velocities(n_particles, target_temperature, mass)
            trajs.append(Atoms(f'H{n_particles}', positions = positions, velocities = velocities, cell = [box_size, box_size, box_size]))
            # Simulation loop
            for step in range(total_steps):
                #track progress and moniter temperature
                print(f"step: {step}", end = '\r')
                # Compute forces
                forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
                forces_repulsive = compute_lennard_jones_forces(positions, epsilon, sigma, box_size, 'repulsive')
                forces_attractive = compute_lennard_jones_forces(positions, epsilon, sigma, box_size, 'attractive')
                total_forces = forces_harmonic + forces_repulsive + forces_attractive
                
                # Integrate equations of motion
                positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass, [box_size, epsilon, sigma, k, r0])
                
                # Apply thermostat
                if step % rescale_interval == 0:
                    velocities = rescale_velocities(velocities, target_temperature, mass)
                
                #save trajectory every 100 steps
                if not (step + 1) % 100:
                    trajs.append(Atoms(f'H{n_particles}', positions = positions, velocities = velocities, cell = [box_size, box_size, box_size]))

                if step + 1 > 10000:
                    #calculate relevant values for steps > 10,000 to get an average
                    Re.append(calculate_end_to_end_distance(positions))
                    Rg.append(calculate_radius_of_gyration(positions))
                    U.append(calculate_potential_energy(positions, box_size, k, r0, epsilon, sigma))

            #write trajectory file
            write(f'trajectories/T-{target_temperature:.2f}_k-{k}_eps-{epsilon["repulsive"]}.extxyz', trajs, 'extxyz')
            end = time.time()
            print(f'runtime: {end-start} seconds')

            #calculateaverages of relevant values
            Rees.append(np.average(Re))
            Rgs.append(np.average(Rg))
            Us.append(np.average(U))

        #write data to csv
        df = pd.DataFrame()
        df['temperature'] = target_temperatures
        df['potential_energy'] = Us
        df['radius_gyration'] = Rgs
        df['end_to_end'] = Rees
        df.to_csv(f'data/k-{k}_eps-{epsilon["repulsive"]}.csv', index=False)