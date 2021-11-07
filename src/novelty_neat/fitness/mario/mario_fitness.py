from typing import List

import numpy as np
from experiments.logger import Logger
from games.level import Level
from games.mario.assets.engine import MarioAstarAgent
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from gym_pcgrl.envs.probs.smb.engine import State
from metrics.solvability import SolvabilityMetric
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.types import LevelNeuralNet


class MarioSolvabilityFitness(NeatFitnessFunction):
    """Returns a solvability score based on if the Mario level is solvable or not. 1 for solvable and 0 for unsolvable.
    """

    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:
        # This should run much faster, because of not having to do the java solv.
        def single_level_solv(level: MarioLevel):
            # lvlString = level.string_representation_of_level()

            # In this case make enemies look like solid tiles, to get around the very hard enemy placement.
            lvlString_enemies = level.string_representation_of_level(
                True).replace("x", '#')
            lvlString_not_enemies = level.string_representation_of_level(False)

            def get_solv_of_string(lvlString):
                state = State()
                state.stringInitialize(lvlString.split("\n"))

                aStarAgent = MarioAstarAgent()

                sol, solState, iters = aStarAgent.getSolution(state, 1, 10000)
                if solState.checkWin():
                    return 1
                sol, solState, iters = aStarAgent.getSolution(state, 0, 10000)
                if solState.checkWin():
                    return 1
                return 0

            return get_solv_of_string(lvlString_enemies) == 1 and get_solv_of_string(lvlString_not_enemies) == 1

        ans = []
        for level_group in levels:
            ans.append(np.mean([single_level_solv(l) for l in level_group]))
        return ans

        metric = SolvabilityMetric(MarioGame(MarioLevel()))
        answer = []
        for level_group in levels:
            answer.append(np.mean(metric.evaluate_mario(level_group)))
        return answer


class MarioNumberEmptyTiles(NeatFitnessFunction):
    """Returns a number from 0 to 1 indicating how many empty tiles there are
    """

    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[MarioLevel]]) -> List[float]:
        answer = []
        for level_group in levels:
            temp = []
            for l in level_group:
                empty_block = l.tile_types_reversed['empty']
                num_empty = (l.map == empty_block).sum()
                num_total = l.map.size
                temp.append(num_empty / num_total)
            answer.append(np.mean(temp))
        return answer


class MarioFeasibilityFitness(NeatFitnessFunction):
    """Returns a number from 0 to 1 indicating how many 'bad' elements there are, specifically, 
        how many enemies and coins aren't straight on top of blocks.
    """

    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[MarioLevel]]) -> List[float]:
        def single(level: MarioLevel):
            curr_score_good = 0
            curr_score_total = 0
            map = level.map
            not_solids = ['empty', 'enemy', 'coin']
            solids = [v for k, v in level.tile_types_reversed.items()
                      if k not in not_solids]
            things_to_check = [
                v for k, v in level.tile_types_reversed.items() if k in not_solids[1:]]
            for ri, row in enumerate(map):
                for ci, cell in enumerate(row):
                    is_enemy_coin = any([t == cell for t in things_to_check])
                    if is_enemy_coin:
                        curr_score_total += 1
                        # check if good
                        if ri == map.shape[0] - 1:
                            # bad, because at very bottom
                            continue
                        right_underneath = map[ri+1, ci]
                        if right_underneath in solids:
                            curr_score_good += 1

            return curr_score_good / curr_score_total if curr_score_total != 0 else 0

        answer = []
        for level_group in levels:
            temp = []
            for l in level_group:
                temp.append(single(l))
            answer.append(np.mean(temp))
        return answer
