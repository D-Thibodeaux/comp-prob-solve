import numpy as np
import matplotlib.pyplot as plt
import MC_simulator_functions as MC 
import random  

kB = 8.617333262E-5
# Parameters based on scenario being tested
scenario_params = {
    "Ideal_Mixture" : {
                        'epsilon_AA': 0.0,#H-H
                        'epsilon_BB': 0.0, #N-N
                        'epsilon_AB': 0.0,
                        }, 
    "Repulsive_Interactions" : {
                        'epsilon_AA': 0.05,#H-H
                        'epsilon_BB': 0.05, #N-N
                        'epsilon_AB': 0.05,
                        }, 
    "Attractive_Interactions" : {
                        'epsilon_AA': -0.05,#H-H
                        'epsilon_BB': -0.05, #N-N
                        'epsilon_AB': -0.05,
                        }, 
    "Immiscible" : {
                        'epsilon_AA': -0.05,#H-H
                        'epsilon_BB': -0.05, #N-N
                        'epsilon_AB': 0.05,
                        }, 
    "Like_Dissolves_Unlike" : {
                        'epsilon_AA': 0.05,#H-H
                        'epsilon_BB': 0.05, #N-N
                        'epsilon_AB': -0.05,
                        }, 
}
def run(scenario, Ts, mus_A, animations, third_species_params = False):
    """
    for a given scenario defining the interactions between particles, run the monte carlo simulation at a number of different values of T and mu_A.

    parameters:
    scenario (str): scenario defining the H-H, H-N, and N-N interactions
    Ts (array) : array of temperature values in units of T/kb 
    mus_A (array) : array of chemical potentials of hydrogen in ev
    animations (array) : array of (T, mu_A) value pairs to animate the MC simulation for
    third_species_params (dict): dictionary of epsilons for a third species

    outputs:
    genrates an image with heatmaps for the mean coverages of H, N and total coverage, as well as an example of the equilibrated lattice at three differet values of (mu_A, T) 
    """
    print(scenario)

    size = 4
    #set epsilons to provided values
    epsilon_AA, epsilon_BB, epsilon_AB = scenario_params[scenario].values()
    if not third_species_params:
        epsilon_C, epsilon_CC, epsilon_AC, epsilon_BC, mu_C = (0,0,0,0,0)  
    else:
        epsilon_C, mu_C = (third_species_params['epsilon_C'], third_species_params['mu_C'])
        epsilon_CC = epsilon_AA
        epsilon_AC,epsilon_BC = (epsilon_AB, epsilon_AB)
    #turn on the third species if parameters are provided
    three = True if third_species_params else False
    n_steps = 10000

    params = []
    for mu_A in mus_A:
        for T in Ts:
            params.append({
                'epsilon_A': -0.1,
                'epsilon_B': -0.1,
                'epsilon_AA': epsilon_AA,#H-H
                'epsilon_BB': epsilon_BB, #N-N
                'epsilon_AB': epsilon_AB,
                'mu_A': mu_A,
                'mu_B': -0.1,
                'T': T,  # Temperature 
                'epsilon_C' : epsilon_C,
                'epsilon_CC' : epsilon_CC,
                "epsilon_AC" : epsilon_AC,
                "epsilon_BC" : epsilon_BC,
                "mu_C" : mu_C
            })

    # Run the simulation
    np.random.seed(42)
    final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
    mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
    mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
    mean_coverage_C = np.zeros((len(mus_A), len(Ts)))

    title = scenario + '-three_species' if third_species_params else scenario
    for i, param in enumerate(params):
        #keep track of progress
        print('\n',i)

        #check and see if the 
        animate = True if (param['T'], param['mu_A']) in animations else False

        lattice, coverage_A, coverage_B, coverage_C = MC.run_simulation(size, n_steps, param,three_species=three, animate=animate, save_every=100, gif_title=f"{title}-{-1*param['mu_A']:.4f}-mu_A-{param['T']:.4f}-T")

        final_lattice[i // len(Ts), i % len(Ts)] = lattice
        mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
        mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])
        mean_coverage_C[i // len(Ts), i % len(Ts)] = np.mean(coverage_C[-1000:])
    # Plot the T-mu_A phase diagram converting temperature to kelvin
    grid = [[0, 1, 6, 2], [3, 4, 5,7]] if third_species_params else [[0, 1, 2], [3, 4, 5]]
    fig, axs = plt.subplot_mosaic(grid, figsize=(6.5/3*len(grid[0]), 4.5))

    # Mean coverage of A 
    axs[0].pcolormesh(mus_A, Ts / kB, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title(r'$\langle \theta_H \rangle$')
    axs[0].set_xlabel(r'$\mu_H (eV)$')
    axs[0].set_ylabel(r'$T (K)$')

    # Mean coverage of B
    axs[1].pcolormesh(mus_A, Ts / kB, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title(r'$\langle \theta_N \rangle$')
    axs[1].set_xlabel(r'$\mu_H (eV)$')
    axs[1].set_yticks([])

    #modify plot for a three species version 
    if three:
        # Mean coverage of C
        axs[6].pcolormesh(mus_A, Ts / kB, mean_coverage_C.T, cmap='viridis', vmin=0, vmax=1)
        axs[6].set_title(r'$\langle \theta_{third} \rangle$')
        axs[6].set_xlabel(r'$\mu_H (eV)$')
        axs[6].set_yticks([])

        # Mean total coverage
        cax = axs[2].pcolormesh(mus_A, Ts / kB, mean_coverage_A.T + mean_coverage_B.T + mean_coverage_C.T, cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title(r'$\langle \theta_H + \theta_N + \theta_{third} \rangle$')
        axs[2].set_xlabel(r'$\mu_H (eV)$')
        axs[2].set_yticks([])
        fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

    else:
        # Mean total coverage
        cax = axs[2].pcolormesh(mus_A, Ts / kB, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
        axs[2].set_xlabel(r'$\mu_H (eV)$')
        axs[2].set_yticks([])
        fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

    # Plot the final lattice configurations for three species
    if three:
            # mu_A = -0.2 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[0, 3], axs[3], r'$\mu_H = -0.2$ eV, $T = 0.01 / k$')

        # mu_A = -0.133 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[2, 3], axs[4], r'$\mu_H = -0.133$ eV, $T = 0.01 / k$')

        # mu_A = -0.067 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[4, 3], axs[5], r'$\mu_H = -0.067$ eV, $T = 0.01 / k$')

        # mu_A = 0 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[6, 3], axs[7], r'$\mu_H = 0$ eV, $T = 0.01 / k$')
    
    # Plot the final lattice configurations for three species
    else:    
        # mu_A = -0.2 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[0, 3], axs[3], r'$\mu_H = -0.2$ eV, $T = 0.01 / k$')

        # mu_A = -0.1 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[3, 3], axs[4], r'$\mu_H = -0.1$ eV, $T = 0.01 / k$')

        # mu_A = 0 eV and T = 0.01 / k
        MC.plot_lattice(final_lattice[6, 3], axs[5], r'$\mu_H = 0$ eV, $T = 0.01 / k$')
        axs[5].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    

    plt.tight_layout()
    plt.savefig(f"images/{title}.png", format='png')

if __name__ == '__main__' : \
    #set up Temperature and mu arrays
    mus_A = np.linspace(-0.2, 0, 7)
    Ts = np.linspace(0.001, 0.019, 7)
    
    #define third species parameters
    third_species = {
        'epsilon_C' : -0.1,
        'mu_C' :-0.1
        }

    for scenario in scenario_params.keys():
        #randomly pick 12 (T,mu_A) value pairs to animate (this value can be changed). 
        #if specific pairs are desired, they can be manually assigned and the random selection will not occur
        #if all combinations should be animated, the value can be set to len(Ts) * len(mus_A) 
        animations = []
        if len(animations) == 0:
            while len(animations) != 12:
                T = random.choice(Ts)
                mu_A = random.choice(mus_A)
                if (T,mu_A) not in animations:
                    animations.append((T,mu_A))

        run(scenario,Ts, mus_A, [], third_species) #run three species version. change [] to the variable animations to make gifs
        #run(scenario,Ts, mus_A, []) #run two species version. change [] to the variable animations to make gifs