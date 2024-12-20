import os 
from glob import glob

files = glob('T-*')
for file in files:
    folder = file.rsplit('_')[1]
    os.system(f'mv {file} {folder}')

for k in [0.5,1.0, 1.5,0.25]:
    for eps_repulsive in [1.0,1.5,2.0,2.5,3.0,4.0]:
        os.system(f'cd k-{k}; mkdir eps-{eps_repulsive}')