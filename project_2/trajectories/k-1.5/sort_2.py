import os 
from glob import glob

files = glob('T-*')
for file in files:
    folder = file.rsplit('_')[2].rsplit('.e')[0]
    os.system(f'mv {file} {folder}')