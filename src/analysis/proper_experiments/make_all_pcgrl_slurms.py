import os


s = \
"""#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J pcgrl_{PCGNN}
#SBATCH -o /home/NAME/PATH_TO_ROOT/src/slurms/logs/all_pcgrl/pcgrl_{GAME}_{REPRESENTATION}_{SEED}.%N.%j.out

source ~/.bashrc
cd /home/NAME/PATH_TO_ROOT/src/external/gym-pcgrl
conda activate pcgrl
echo "Hello, we are doing a job now, {GAME}, {REPRESENTATION}, with seed =  {SEED}"
./run.sh train.py {GAME} {REPRESENTATION} {SEED}
"""

for game in ['smb', 'binary']:
    for repr in ['turtle', 'wide']:
        for seed in range(1, 6):
            name = f'{game}-{repr}-{seed}'
            K = s.format(GAME=game, REPRESENTATION=repr, SEED=seed, PCGNN=name)
            dir = 'slurms/all_pcgrl'
            os.makedirs(dir, exist_ok=True)
            with open(os.path.join(dir, f'{name}.batch'), 'w+') as f:
                f.write(K)