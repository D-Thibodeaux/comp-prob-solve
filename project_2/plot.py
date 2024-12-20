import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os
#get all data files
files = glob('data/*')

#make plots for each file 
for file in files:
    #read file
    df = pd.read_csv(file)

    location = f'images/k-{file.rsplit('-')[1].rsplit('_')[0]}/eps-{file.rsplit('-')[2].rsplit('.c')[0]}'
    #make folder to better organize images if they don't already exist
    if len(glob(location)) == 0:

        os.system(f'cd {location.rsplit('/')[0]}; mkdir {location.rsplit('/')[1]}')
        os.system(f'cd {location.rsplit('/e')[0]}; mkdir {location.rsplit('/')[2]}')

    #plot potential energy
    fig = plt.figure(1,figsize=(6,6))
    plt.plot(df['temperature'], df['potential_energy'])
    plt.title('Potential energy as a function of temperature')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Potential Energy (eV)')
    plt.tight_layout
    plt.savefig(f'{location}/potential_energy-{file.rsplit('.c')[0].rsplit('/')[1]}.png', format = 'png')
    plt.close()

    #plot end to end distance
    fig = plt.figure(2,figsize=(6,6))
    plt.plot(df['temperature'], df['end_to_end'])
    plt.title('End to end distance as a function of temperature')
    plt.xlabel('Temperature (K)')
    plt.ylabel('distance')
    plt.tight_layout
    plt.savefig(f'{location}/end_to_end-{file.rsplit('.c')[0].rsplit('/')[1]}.png', format = 'png')
    plt.close()

    #plot radius of gyration
    fig = plt.figure(3,figsize=(6,6))
    plt.plot(df['temperature'], df['radius_gyration'])
    plt.title('Radius of gyration as a function of temperature')
    plt.xlabel('Temperature (K)')
    plt.ylabel('radius')
    plt.tight_layout
    plt.savefig(f'{location}/radius_of_gyration-{file.rsplit('.c')[0].rsplit('/')[1]}.png', format = 'png')
    plt.close()