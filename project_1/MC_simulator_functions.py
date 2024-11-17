import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from glob import glob 
from PIL import Image as Image
import moviepy.editor as mp

def initialize_lattice(size):
    """
    generate a square lattice of a provided size

    Parameters:
    size (int) : number of lattice sites along one side of the overall grid

    outputs:
    size x size lattice filled with zeros
    """
    lattice = np.zeros((size,size))
    return lattice

def compute_neighbor_indices(size):
    """
    for a given sized square lattice, determines the four neighbors of each index pair

    Parameters:
    size (int) : number of lattice sites along one side of the overall grid

    outputs:
    dictionary with index pairs as keys and the associated neighbors as values
    """
    neighbor_indices = {}
    #loop over all x indicies
    for x in range(size):
        #loop over all y indicies
        for y in range(size):
            #for a given x,y pair, determine the 4 nearest neighbors. using % size accounts for periodic boundaries by wrapping indicies at the edge
            #EX (for the the index 15 and size 16): (x+1)%size =  16%16 = 0
            neighbors = [((x - 1) % size, y), ((x + 1) % size, y), (x, (y - 1) % size), (x, (y + 1) % size)]
            #store the list of neighbors for each (x,y) pair 
            neighbor_indices[(x, y)] = neighbors
    #return neighbor pair dictionary
    return neighbor_indices

def calculate_interaction_energy(lattice, site, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB, epsilon_CC, epsilon_AC, epsilon_BC):
    """
    calculate the interaction energy of a particle at a given site on the lattice

    parameters:
    lattice (array) : matrix containing information about the occupancy of each site
    site (array) : indicies of the site being considered in the form (x,y)
    neighbor_indices (dict) : dictionary containing the indicies of the neigbors for all posible sites
    epsilon_AA (float) : interaction energy for two neighboring A particles 
    epsilon_BB (float) : interaction energy for two neighboring B particles 
    epsilon_AB (float) : interaction energy for neighboring A and B particles 
    epsilon_CC (float) : interaction energy for two neighboring C particles.
    epsilon_AC (float) : interaction energy for neighboring A and C particles.
    epsilon_BA (float) : interaction energy for neighboring B and C particles. 

    outputs:
    interaction energy as a float in whatever units the epsilon values were provided in 
    """
    x, y = site
    particle = lattice[x,y]
    E_total = 0
    #loop over the 4 neighbors of the site
    for neighbor_pos in neighbor_indices[(x, y)]:
        neighbor = lattice[neighbor_pos]
        
        #calculate the interaction energy between the particle and the neighbor 
        if particle == 1:  # Particle A
            if neighbor == 2:
                E_total += epsilon_AA
            elif neighbor == 1:
                E_total += epsilon_AB
            elif neighbor == 3:
                E_total += epsilon_AC
            
        elif particle == 2:  # Particle B
            if neighbor == 2:
                E_total += epsilon_BB
            elif neighbor == 1:
                E_total += epsilon_AB
            elif neighbor == 3:
                E_total += epsilon_BC
        
        elif particle == 3:  # Particle C
            if neighbor == 3:
                E_total += epsilon_CC
            elif neighbor == 1:
                E_total += epsilon_AC
            elif neighbor == 2:
                E_total += epsilon_BC
    return E_total

def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params, N_C, three_species = False):
    """
    randomly attempts to add or remove a particle to the lattice based on the binding and interaction energies

    parameters:
    lattice (array) : matrix containing information about the occupancy of each site
    N_A (int) : number of particles of type A
    N_B (int) : number of particles of type B
    N_empty (int) : number of empty sites
    neighbor_indices (dict) : dictionary containing the indicies of the neigbors for all posible sites
    params (array) : array containing temperature, epsilon values, and chemical potentials
    N_C (int) : number of particles of type C
    three_species (bool) : if True allows third particle type to be added to surface by random moves

    outputs:
    number of particles of each species and number of empty sites  
    """
    size = len(lattice)
    N_sites = size * size
    epsilon_A, epsilon_B, epsilon_AA, epsilon_BB, epsilon_AB, mu_A, mu_B, T, epsilon_C, epsilon_CC, epsilon_AC, epsilon_BC, mu_C = params.values()
    beta = 1 / T
    
    #list with all possible particles, adds an extra option if doing a three species simulation
    particle_types = [1,2]
    if three_species:
        particle_types.append(3)
        
    add = np.round(np.random.rand())

    if add:
        if N_empty == 0:
            return N_A, N_B, N_C, N_empty
        
        #guess a random site, check if it is empty, reject and pick a new site if occupied, continue if site is unoccupied
        found = False
        while not found:
            site = (np.random.choice(range(size)), np.random.choice(range(size)))
            if lattice[site] == 0: 
                found = True
        particle = np.random.choice(particle_types)
        if particle == 1:
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A

        elif particle == 2:
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B

        elif particle == 3:
            mu = mu_C
            epsilon = epsilon_C
            N_s = N_C

        delta_E = epsilon + calculate_interaction_energy(lattice, site, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB, epsilon_CC, epsilon_AC, epsilon_BC)
        acc_prob = np.min([1, (N_empty) / (N_s + 1) * np.exp(-beta * (delta_E - mu))])
        r = np.random.rand()
        if r < acc_prob:
            lattice[site] = particle
            if particle == 1:
                N_A += 1
            elif particle == 2:
                N_B += 1
            elif particle == 3:
                N_C += 1
            N_empty -= 1
    #remove particle 
    else:
        #if all sites are empty return particle numbers
        if N_empty == N_sites:
            return N_A, N_B, N_C, N_empty
        
        #guess a random site, check if it is occupied, reject and pick a new site if empty, continue if site is occupied
        found = False
        while not found:
            site = (np.random.choice(range(size)), np.random.choice(range(size)))
            if lattice[site] != 0: 
                found = True
        
        particle = lattice[site]

        if particle == 1:
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A

        elif particle == 2:
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B

        elif particle == 3:
            mu = mu_C
            epsilon = epsilon_C
            N_s = N_C

        delta_E = -epsilon - calculate_interaction_energy(lattice, site, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB, epsilon_CC, epsilon_AC, epsilon_BC)

        acc_prob = np.min([1, N_s / (N_empty + 1) * np.exp(-beta * (delta_E + mu))])
        r = np.random.rand()
        if r < acc_prob:
            lattice[site] = 0
            if particle == 1:
                N_A -= 1
            elif particle == 2:
                N_B -= 1
            elif particle == 3:
                N_C -= 1
            N_empty += 1

    return N_A, N_B, N_C, N_empty

def plot_lattice(lattice, ax, title):
    """
    plot provided lattice for visualization purposes

    parameters:
    lattice (array) : matrix containing information about the occupancy of each site
    ax (matplotlib axis object) : axis on which to display the lattice
    title (str) : title of the plotted lattice

    outputs:
    atplotlib axis object with the displayed lattice
    """
    size = len(lattice)
    for x in range(size):
        for y in range(size):
            if lattice[x,y] == 1:
                ax.scatter(x + 0.5, y + 0.5, color = 'red')
            if lattice[x,y] == 2:
                ax.scatter(x + 0.5, y + 0.5, color = 'blue')
            if lattice[x,y] == 3:
                ax.scatter(x + 0.5, y + 0.5, color = 'green')

    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_xticks(np.arange(0, size, 1))
    ax.grid(which='major')
    ax.tick_params(axis = 'both', which = 'both', labelbottom = False, labelleft = False)
    ax.set_title(title, fontsize=10)

    return ax

def make_animation(label):
    """
    animate the lattice model using the images created during the Monte Carlo simulation.

    parameters:
    label (str) : title of the animation file

    output:
    creates a gif with provided title
    """

    holding = []
    for item in glob("temporary_image_storage/*"):
        holding.append((int(item.rsplit('/')[1].rsplit('.')[0]), item))

    holding = sorted(holding)

    images = []
    for i, item in holding:
        images.append(Image.open(item))

    #make gif with images created by the 
    images[0].save('gifs/' + label + '.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, disposal = 2, loop=0)
    #remove images used to make gif
    os.system("rm temporary_image_storage/*")

def run_simulation(size, n_steps, params, three_species = False, animate = False, save_every = 1, gif_title = ''):
    """
    Run the monte carlo simulation 

    parameters:
    size (int) : number of lattice sites along one side of the overall grid
    n_steps (int): number of steps in the simulation
    params (array) : array containing temperature, epsilon values, and chemical potentials
    three_species (bool) : if True allows third particle type to be added to surface by random moves
    animate (bool) : if True plots the lattice at each step and turns it into a video for viewing
    save_every (int) : number of steps to take before saving configuration
    gif_title (str) : title for gif if 

    outputs:
    final configuration of the lattice after all steps have been completed, coverage of each species at each step
    """
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size
    
    N_A = 0
    N_B = 0
    N_C = 0
    N_empty = N_sites

    coverage_A = np.zeros(n_steps)
    coverage_B = np.zeros(n_steps)
    coverage_C = np.zeros(n_steps)

    for step in range(n_steps):
        #keep track of progress
        print(step + 1, end = '\r')
        N_A, N_B, N_C, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params, N_C = N_C, three_species = three_species)
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites
        coverage_C[step] = N_C / N_sites
        if animate and step % save_every == 0:
            fig, ax = plt.subplots()
            ax = plot_lattice(lattice, ax, r'$\mu_A = {:.2f}$ eV, $T = {:.2f} / k$, step {}'.format(params['mu_A'], params['T'],step))
            plt.savefig(f"temporary_image_storage/{step}.png", format = 'png')
            plt.close()
    if animate:
        make_animation(gif_title)
    return lattice, coverage_A, coverage_B, coverage_C