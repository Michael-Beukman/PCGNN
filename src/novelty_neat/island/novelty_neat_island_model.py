import copy
from math import ceil
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import ray
from common.methods.pcg_method import PCGMethod
from experiments.logger import Logger, NoneLogger
from games.game import Game
from games.level import Level
from novelty_neat.novelty_neat import NoveltyNeatPCG
import neat


class NoveltyNeatIslandModel(PCGMethod):
    def __init__(self, game: Game,
                 init_level: Level,
                 islands: List[NoveltyNeatPCG],
                 number_of_migration_steps: int = 10) -> None:
        """This makes an island model of two or more populations, similarly to this method here:
            Whitley, D., Rana, S., & Heckendorn, R. B. (1999). The island model genetic algorithm: On separability, population size and convergence. Journal of computing and information technology, 7(1), 33-47.

        Args:
            game (Game): The game
            init_level (Level): The initial level, not really used
            islands (List[NoveltyNeatPCG]): The 2 or more different methods that need to be combined.
                These methods must have the same network structure, but otherwise they can differ.
            number_of_migration_steps (int): How many times do we call model.train() and swap the populations.
        """
        super().__init__(game, init_level)
        self.islands = islands
        self.number_of_migration_steps = number_of_migration_steps
        assert len(
            self.islands) >= 2, "Island model requires at least 2 populations"
        all_pop_sizes = [len(i.pop.population) for i in self.islands]
        # assert the pop sizes are the same.
        assert min(all_pop_sizes) == max(all_pop_sizes)
        self.best_agent: Union[neat.DefaultGenome, None] = None
        self.best_fitness = 0;
        self.best_pop = None

    def train(self, logger: Logger) -> List[Dict[str, Any]]:
        five_percent = ceil(len(self.islands[0].pop.population) / 20)

        def fitness(which_neat_pop: NoveltyNeatPCG, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config):
            ans = {}
            nets = []
            for genome_id, genome in genomes:
                nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
            all_fitnesses = which_neat_pop.fitness_calculator(nets)
            for fit, (id, genome) in zip(all_fitnesses, genomes):
                genome.fitness = fit
                ans[id] = fit
            return ans

        def _get_values(pop: NoveltyNeatPCG, pops: dict, config: neat.Config, name: str):
            fits = fitness(pop, pops.items(), config)
            list_fits = list(fits.values())
            print(
                f"Pop {name}, min = {np.min(list_fits)}, max = {np.max(list_fits)}, mean = {np.mean(list_fits)} ")
            fit_indices = sorted(
                map(tuple, map(reversed, fits.items())), reverse=True)
            fit_indices = [(id, pops[id]) for (_, id) in fit_indices]

            top_5 = fit_indices[:five_percent]
            # remove bottom 5
            bad_5 = fit_indices[-five_percent:]
            fit_indices = fit_indices[:-five_percent]
            

            # set my best value.
            most_fit_id_in_this_batch = fit_indices[0][0]
            if fits[most_fit_id_in_this_batch] > self.best_fitness:
                self.best_fitness = fits[most_fit_id_in_this_batch]
                self.best_agent = copy.deepcopy(pops[most_fit_id_in_this_batch])
                self.best_pop = pop

            return fit_indices, top_5, bad_5

        def update_fitnesses(fit_indices, bad_this, top_other):
            fit_indices.extend([
                (id_bad, fit) for ((id_bad, bad_fit), (id_og, fit)) in zip(bad_this, top_other)
            ])
            return fit_indices
        jump_size = 1
        
        @ray.remote
        def train_pop(pop):
            pop.train(NoneLogger())
            return pop

        for i in range(self.number_of_migration_steps):
            print(f"MIGRATION STEP {i+1} / {self.number_of_migration_steps}. LEN = {len(self.islands)}")
            self.best_fitness = 0
            # train

            # faster stuff
            vals = [train_pop.remote(p) for p in self.islands]
            new_pops = ray.get(vals)
            self.islands = new_pops
            # for island in self.islands:island.train(logger)

            # now calculate fitnesses.
            all_values = []
            for j, pop in enumerate(self.islands):
                fit_indices_j, top_5_j, bad_5_j = _get_values(
                    pop, pop.pop.population, pop.neat_config, f'pop_{j}')
                all_values.append((fit_indices_j, top_5_j, bad_5_j))

            all_fit_indices = []
            for j in range(len(self.islands)):
                current_value = all_values[j]
                new_j = (j + jump_size) % len(self.islands)
                assert j != new_j
                next_value = all_values[new_j]

                fit_indices_now = update_fitnesses(
                    current_value[0], next_value[-1], current_value[1])
                all_fit_indices.append(fit_indices_now)

            # now we actually update
            for j, pop in enumerate(self.islands):
                pop.pop.population = {
                    id: genome for id, genome in all_fit_indices[j]
                }
                for K in range(len(pop.fitness_calculator.fitnesses)):
                    if hasattr(pop.fitness_calculator.fitnesses[K], 'previously_novel_individuals') and len(pop.fitness_calculator.fitnesses[K].previously_novel_individuals) >= 100:
                        # clean up the novelty archive, otherwise it takes way too long.
                        pop.fitness_calculator.fitnesses[K].previously_novel_individuals = []

            jump_size += 1
            # already wrapped around
            if jump_size % len(self.islands) == 0:
                jump_size = 1
        return [
            {
                'populations': self.islands
            }
        ]

    def generate_level(self) -> Level:
        return self.best_pop.level_generator(neat.nn.FeedForwardNetwork.create(self.best_agent, self.best_pop.neat_config))

    def __repr__(self) -> str:
        return f"NoveltyNeatIslandModel(number_of_migration_steps={self.number_of_migration_steps})"