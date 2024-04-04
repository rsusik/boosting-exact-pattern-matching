import subprocess, os
from tqdm import tqdm

for filename in tqdm(os.listdir('source/algos/')):
    if not filename.endswith('.c'): continue
    
    out = subprocess.check_output(
        f'gcc source/algos/{filename} -O3 -msse4 -lm -w -o source/bin/{filename[:-2]}'.split(), 
#         stderr=open(os.devnull, 'wb')
    )
    if out != b'':
        print('ERROR', filename, out)
        