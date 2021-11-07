# Maze
## Generalisation

### 104 - NEAT - Larger Sizes
cd v100 && sbatch v104.batch && cd -

### 107 - DirectGA - Larger Sizes
cd v100 && sbatch v107.batch && cd -


## Methods
## DirectGA+ - 102aaa
cd v100 && sbatch v102_0.batch && cd -

## DirectGA (Novelty) - 102f
cd v100 && sbatch v102_1.batch && cd -

## NEAT - 105
cd v100 && sbatch v105.batch && cd -

# Mario

## Generalisation
### 206 - NEAT + DirectGA - Larger Sizes
cd v200 && sbatch v206_0.batch && cd -
cd v200 && sbatch v206_1.batch && cd -
cd v200 && sbatch v206_2.batch && cd -


## Methods

## DirectGA - 201b
cd v200 && sbatch v201_1.batch && cd -

## DirectGA+ - 201a
cd v200 && sbatch v201_0.batch && cd -

## DirectGA (Novelty) - 201d
cd v200 && sbatch v201_2.batch && cd -

## NEAT - 204e
cd v200 && sbatch v204.batch && cd -



# Metrics
cd v200 && sbatch v202.batch && cd -

cd v100 && sbatch v106.batch && cd -

# v108:
cd v100/v108 && bash run_all.sh && cd -