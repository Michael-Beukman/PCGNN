import pickle
import neat
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel

from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling

def generate_level(game: str, width: int = 14, height: int = 14):
    if game.lower() == 'maze':
        pickle_path = '../results/experiments/experiment_105_a/Maze/NEAT/2021-10-31_13-23-23/50/200/4/seed_4_name_experiment_105_a_2021-10-31_13-40-59.p'
        config = './runs/proper_experiments/v100_maze/config/tiling_generate_12_1_balanced_pop100'
        with open(pickle_path, 'rb') as f:
            dic = pickle.load(f)
        
        best = dic['train_results'][0]['final_agent']

        neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config)
        net = neat.nn.FeedForwardNetwork.create(best, neat_conf)
        generator = GenerateMazeLevelsUsingTiling(game=MazeGame(MazeLevel(width, height)), number_of_random_variables=4, 
                should_add_coords=False,
                do_padding_randomly=False,
                should_start_with_full_level=False, 
                random_perturb_size=0.1565)
    else:
        pickle_path = '../results/experiments/experiment_204e/Mario/NeatNovelty/2021-10-25_20-04-31/100/150/True/1/0/6/2/-1/True/0/seed_0_name_experiment_204e_2021-10-25_23-53-36.p'
        with open(pickle_path, 'rb') as f:
            dic = pickle.load(f)
        
        best = dic['train_results'][0]['final_agent']
        config = './novelty_neat/configs/tiling_mario_56_7_1pred_size_one_hot_100_pop_clean'

        neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config)
        net = neat.nn.FeedForwardNetwork.create(best, neat_conf)
        generator = GenerateGeneralLevelUsingTiling(MarioGame(MarioLevel(width, height)), 1, 4, False, 0, 
                                                    predict_size=1, 
                                                    reversed_direction = 0, 
                                                    use_one_hot_encoding=True)
        
    level = generator(net)

    return level