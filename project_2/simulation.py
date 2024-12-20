import numpy as np

def apply_pbc(position, box_size):
    """
    wrap positions if they are outside of the cubic box
    
    parameters:
    position (array) : three item array containing the position of a particle in (x,y,z)
    box_size (float) : side length of the cubic simulation box

    outputs: 
    position with periodic boundary position applied
    """
    return position % box_size

def minimum_image(displacement, box_size):
    """
    calculate the minimum distace between two particles considering peiodic boundaries

    parameters:
    displacement (array): three item array containing the displacement between two particles in all three directions
    box_size (float) : side length of the cubic simulation box

    outputs: 
    minimum image displacement
    """
    #decrease displacement by one box length in x,y,and/or z if displacement is larger than 0.5 box_length
    disp = displacement - box_size * np.round(displacement / box_size)
    return disp

def initialize_chain(n_particles, box_size, r0):
    """
    generate a chain of non-overlaping particles to begin the simulation

    paramters:
    n_particles (int) : number of particles in the polymer chain
    box_size (float) : side length of the cubic simulation box
    r0 (float) : eqilibrium distance between two particles

    output:
    array with the positions of all particles in the polymer chain
    """
    #generate empty array with a length of n_particles 
    positions = np.zeros((n_particles, 3))
    #start the chain at the center of the box
    current_position = [box_size/2, box_size/2, box_size/2]
    positions[0] = current_position
    #build the rest of the chain
    for i in range(1,n_particles):
        #generate random 3D vector
        direction = np.random.rand(3)
        #divide by magnitude to make unit vector
        direction /= np.linalg.norm(direction)
        #put a new particle a distance of r0 away
        next_position = current_position + r0 * direction
        positions[i] = apply_pbc(next_position, box_size)
        current_position = positions[i]
    return positions

def initialize_velocities(n_particles, target_temperature, mass):
    """
    give all the particles random velocites that obey the maxwell boltzmann distribution for the provided target temperature

    parameters:
    n_particles (int) : number of particles in the polymer chain
    target_temperature (float) : temperature the simulation should occur at in units of T/kB
    mass (float) : mass of each particle

    output:
    array of velocities with length of n_particles
    """
    #assign random velocities from maxwell boltzmann distributuion
    velocities = np.random.normal(0, np.sqrt(target_temperature / mass), (n_particles, 3))
    velocities -= np.mean(velocities)  # Remove net momentum
    return velocities

def compute_harmonic_forces(positions, k, r0, box_size):
    """
    calculate the harmonic force on each particle

    parameters:
    positions (array) : array containing the position of each particle in the chain
    k (float) : spring constant
    r0 (float) : equilibrium distance between two particles
    box_size (float) : side length of the cubic simulation box

    outputs:
    array of harmonic forces corresponding to the particle array
    """
    #create empty numpy array with the same dimensions as the positions array
    forces = np.zeros(positions.shape)
    #calculate forces
    for i in range(len(positions) - 1):
        #calculate minimum image distance
        displacement = positions[i+1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        #calculate force
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        #apply forces equally on the two particles with opposite directions
        forces[i] -= force
        forces[i+1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    """
    compute the attractive or repulsive lennard jones forces

    parameters:
    positions (array) : array containing the position of each particle in the chain
    epsilon (dictionary) : dictionary containing attractive and repulsive epsilon values
    sigma (float) : sigma vlaue for leard jones equation
    box_size (float) : side length of the cubic simulation box
    interaction_type (string) : string specifying the type of interaction being calculated

    output:
    array of forces from the lennard jones potential
    """
    forces = np.zeros(positions.shape)
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if interaction_type == 'repulsive' and np.abs(i - j) == 2:
                eps = epsilon['repulsive']
            elif interaction_type == 'attractive' and np.abs(i-j) > 2:
                eps = epsilon['attractive']
            else:
                continue
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)

            if distance < 3 * sigma:
                force_magnitude = 24 * eps * ((sigma/distance)**12 - 0.5 * (sigma / distance)**6)  / distance
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass, parameters):
    """
    caluclate new positions and velocities using velocity verlet algorithm

    parameters:
    positions (array) : array containing the positions of all particles
    velocities (array) : array containing the velocities of all particles
    forces (array) : array containing the forces on each particle
    dt (float) : timestep used in simulation
    mass (float) : mass of each particle
    parameters (list) : list containing the values box_size, epsilon, sigma, k, r0 in that order

    output:
    three arrays containing the new vaules of position, velocity and forces
    """
    box_size, epsilon, sigma, k, r0 = parameters
    new_velocities = velocities + 0.5 * forces / mass * dt
    new_positions = positions + new_velocities * dt
    new_positions = apply_pbc(new_positions, box_size)
    forces_new = compute_harmonic_forces(positions, k, r0, box_size) + compute_lennard_jones_forces(positions, epsilon, sigma, box_size, 'repulsive') + compute_lennard_jones_forces(positions, epsilon, sigma, box_size, 'attractive')
    new_velocities += 0.5 * forces_new / mass * dt
    return new_positions, new_velocities, forces_new

def rescale_velocities(velocities, target_temperature, mass):
    """
    rescale the velocities to be consitant with the maxwell boltzmann distribution

    parameters: 
    velocities (array) : arracy containing the velocities of all the particles
    target_temperature (float) : temperature the simulation takes place at in units of T/kB
    mass (float) : mass of each particle

    output:
    array containing rescaled velocities
    """
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1)**2)
    current_temperature = (2/3) * kinetic_energy / (len(velocities))
    #print(current_temperature)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    scaled_velocities = scaling_factor * velocities
    return scaled_velocities