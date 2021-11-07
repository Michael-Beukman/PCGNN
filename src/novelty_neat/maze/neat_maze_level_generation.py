from re import L
import numpy as np
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.types import LevelNeuralNet
from games.level import Level
import neat

class GenerateMazeLevelUsingOnePass(NeatLevelGenerator):
    """This generates a maze level from just the output of the network, i.e. the network must return 
        a list of length width * height.
    """
    def __init__(self, game: MazeGame, number_of_random_variables: int = 2):
        super().__init__(number_of_random_variables)
        self.game = game

    def generate_maze_level_using_one_pass(self, input: np.ndarray, net: LevelNeuralNet) -> Level:
        """Performs a single forward pass and uses that as the level (after some reshaping)

        Args:
            input (np.ndarray): The random inputs
            net (LevelNeuralNet): The network that generates the levels.

        Returns:
            Level: The generated level.
        """
        outputs = np.array(net.activate(input))
        total_number_of_elements_expected = self.game.level.width * self.game.level.height
        assert outputs.shape == (total_number_of_elements_expected,), f"Shape of One pass output should be {total_number_of_elements_expected}"
        return MazeLevel.from_map((outputs.reshape((self.game.level.height, self.game.level.width)) > 0.5).astype(np.int32))

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        return self.generate_maze_level_using_one_pass(input, net)

class GenerateMazeLevelsUsingTiling(NeatLevelGenerator):
    def __init__(self, game: MazeGame, tile_size: int = 1, number_of_random_variables: int = 2, 
                  should_add_coords: bool = False,
                  do_padding_randomly: bool = False,
                  should_start_with_full_level: bool = False,
                  random_perturb_size: float = 0,
                  do_empty_start_goal: bool = False,
                  reverse: int = 0
                  ):
        """Generates levels using a tiling approach, i.e. moving through all of the cells, giving the network the surrounding ones and taking the prediction as the current tile.

        Args:
            game (MazeGame): 
            tile_size (int, optional): Not used. Must be 1. Defaults to 1.
            number_of_random_variables (int, optional): The number of random variables to add to the network. Defaults to 2.
            should_add_coords (bool, optional): If this is true, we append the normalised coordinates of the current cell to the input to the network. Defaults to False.
            do_padding_randomly (bool, optional): If this is true, then we don't pad with -1s around the borders, but we instead make those random as well. Defaults to False.
            should_start_with_full_level (bool, optional): The initial level, instead of being random, is completely filled with 1s. Defaults to False.
            random_perturb_size (float, optional): If this is nonzero, all inputs to the net, including coordinates and surrounding tiles will be randomly perturbed by a gaussian (mean 0, variance 1) 
                multiplied by this value. Defaults to 0.
            do_empty_start_goal (bool, optional): If True, then we make the start and end positions empty, no matter what the network predicts. Defaults to False
            reverse: 0 -> normal
                     1 -> iterate from bottom right to top left instead of other way around
        """
        super().__init__(number_of_random_variables)
        self.game = game
        self.subtile_width = tile_size
        self.tile_size = 3 # tile_size
        self.should_add_coords = should_add_coords
        self.do_padding_randomly = do_padding_randomly
        self.should_start_with_full_level = should_start_with_full_level
        self.random_perturb_size = random_perturb_size
        self.do_empty_start_goal = do_empty_start_goal
        self.reverse = reverse
        assert self.tile_size == 3, "Not supported for different tiles sizes yet."
    
    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        return self.generate_maze_level_using_tiling(input, net)

    def generate_maze_level_using_tiling(self, input: np.ndarray, net: LevelNeuralNet) -> Level:
        """What we want to do here is to generate levels incrementally, i.e. we give the network 10 inputs,
            8 adjacent tiles and the given random numbers, then we ask it to predict the current tile.
            Ideally we would want subtiles to be predicted, but I won't do that now.
        Args:
            input (np.ndarray): The random numbers that act as input
            net (LevelNeuralNet): The network that actually generates the level. 
                Must take in len(input) + self.tile_size ** 2 - 1 numbers and output a single one.

        Returns:
            Level: 
        """
        h, w = self.game.level.height, self.game.level.width
        half_tile = self.tile_size // 2
        # net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = 1.0 * (np.random.rand(h + 2 * half_tile, w + 2 * half_tile) > 0.5)
        else:
            output = np.zeros((h + half_tile * 2, w + half_tile * 2)) - 1 # pad it
            if self.should_start_with_full_level:
                output[half_tile:-half_tile, half_tile:-half_tile] = np.ones((h, w)) # initial level
            else:
                output[half_tile:-half_tile, half_tile:-half_tile] = 1.0 * (np.random.rand(h, w) > 0.5) # initial level
        input_list = list(input)
        # assert output.sum() != 0
        
        row_range = range(half_tile, h + half_tile)
        col_range = range(half_tile, w + half_tile)
        if self.reverse == 1:
            row_range = reversed(row_range)
            col_range = reversed(col_range)
        
        # This is super important, as the reversed thing is a one use iterator!
        # You cannot iterate multiple times!!!!!!!!!!
        row_range = list(row_range)
        col_range = list(col_range)

        for row in row_range:
            for col in col_range:
                # get state
                little_slice = output[row - half_tile: row + half_tile + 1, col - half_tile: col + half_tile + 1]
                # This should be a 3x3 slice now.
                assert little_slice.shape == (self.tile_size, self.tile_size)
                total = self.tile_size * self.tile_size
                little_slice = little_slice.flatten()
                
                # Remove the middle element, which corresponds to the current cell.
                little_slice_list = list(little_slice)
                little_slice_list.pop(total//2)
                assert len(little_slice_list) == total - 1, f"{len(little_slice)} != {total-1}"
                
                # Add in random input.
                little_slice_list.extend(input_list)
                
                if self.should_add_coords:
                    # Normalised coords between 0 and 1.
                    little_slice_list.extend([
                        (row - half_tile) / (h - 1),
                        (col - half_tile) / (w - 1)
                    ])
                
                input_to_net = little_slice_list
                assert len(input_to_net) == total -1 + self.number_of_random_variables + self.should_add_coords * 2
                if self.random_perturb_size != 0:
                    # Perturb input randomly.
                    input_to_net = np.add(input_to_net, np.random.randn(len(input_to_net)) * self.random_perturb_size)
                
                output_tile = net.activate(input_to_net)[0]
                # Threshold
                output[row, col] = (output_tile > 0.5) * 1.0

                if self.do_empty_start_goal:
                    # If we empty the start and end, and we are either at the goal or start, make this tile 0.
                    if row == half_tile and col == half_tile or row == h + half_tile - 1 and col == w + half_tile - 1:
                        output[row, col] = 0
        
        thresh = 0.5
        # if np.any(output < -0.1): thresh = 0
        # Take only relevant parts.
        output = output[half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (h, w)
        return MazeLevel.from_map((output > thresh).astype(np.int32))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, should_add_coords={self.should_add_coords}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size}, do_empty_start_goal={self.do_empty_start_goal}, reverse={self.reverse})"

class GenerateMazeLevelsUsingTilingVariableTileSize(NeatLevelGenerator):
    def __init__(self, game: MazeGame, tile_size: int = 1, number_of_random_variables: int = 2,
                  do_padding_randomly: bool = False,
                  random_perturb_size: float = 0):
        super().__init__(number_of_random_variables)
        self.game = game
        self.subtile_width = tile_size
        self.tile_size = 3 # tile_size
        assert self.tile_size == 3, "Not supported for different tiles sizes yet."
        self.do_padding_randomly = do_padding_randomly
        self.random_perturb_size = random_perturb_size
    
    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        return self.generate_maze_level_using_tiling_bigger_sizes(input, net)

    def generate_maze_level_using_tiling_bigger_sizes(self, input: np.ndarray, net: LevelNeuralNet) -> Level:
        """This should generate levels on the same principles, but we should take bigger tiles. Thus, to predict the following:

            a b    e f    i j
            c d    g h    k l

            m n    x y    q r
            o p    z w    s t

            1 2    5 6    9 A
            3 4    7 8    B C

        If we want to predict the 4 tiles x, y, z, w, we give the network all the shown tiles as context (even x, y, z, w).

        Args:
            input (np.ndarray): [description]
            net (LevelNeuralNet): [description]

        Returns:
            Level: [description]
        """

        size = self.subtile_width

        h, w = self.game.level.height, self.game.level.width
        half_tile = self.tile_size // 2 * size
        
        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = 1.0 * (np.random.rand(h + 2 * half_tile, w + 2 * half_tile) > 0.5)
        else:
            output = np.zeros((h + half_tile * 2, w + half_tile * 2)) - 1 # pad it
            output[half_tile:-half_tile, half_tile:-half_tile] = 1.0 * (np.random.rand(h, w) > 0.5) # initial level

        input_list = list(input)
        assert output[half_tile:-half_tile, half_tile:-half_tile].sum() != 0
        output[half_tile:-half_tile, half_tile:-half_tile] = 1*(output[half_tile:-half_tile, half_tile:-half_tile] > 0.5)
        for row in range(half_tile, h + half_tile, size):
            for col in range(half_tile, w + half_tile, size):
                # little_slice = output[row - half_tile: row + half_tile + 1, col - half_tile: col + half_tile + 1]
                little_slice = output[row - size: row + size * 2, col - size: col + size * 2]

                assert little_slice.shape == (3 * size, 3 * size)
                little_slice_list = list(little_slice.flatten())
                little_slice_list.extend(input_list)
                
                input_to_net = little_slice_list
                if self.random_perturb_size != 0:
                    # Perturb input randomly.
                    input_to_net = np.add(input_to_net, np.random.randn(len(input_to_net)) * self.random_perturb_size)
                
                output_tiles = np.array(net.activate(input_to_net)).reshape(size, size)
                output[row: row + size, col: col + size] = output_tiles > 0.5
        
        output = output[half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (h, w)
        return MazeLevel.from_map((output > 0.5).astype(np.int32))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size})"

class GenerateMazeLevelsUsingCPPNCoordinates(NeatLevelGenerator):
    """Generates a level from a CPPN using the coordinates of the cells as well as a random input.
        Thus, we input (x, y, r1, r2, ..., rn) and get out a tile type.

    """
    def __init__(self, game: MazeGame, number_of_random_variables: int, new_random_at_each_step:bool = False):
        """
        Args:
            game (MazeGame): The game to generate levels for
            number_of_random_variables (int): How many random numbers need to be generated for one pass through the network.
            new_random_at_each_step (bool, optional): If this is true, then we generate a new random number (or numbers) 
                for each cell we generate. Defaults to False.
        """
        super().__init__(number_of_random_variables=number_of_random_variables)
        self.game = game
        self.new_random_at_each_step = new_random_at_each_step

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        h, w = self.game.level.height, self.game.level.width
        output = np.zeros((h, w))
        random_list = list(input)
        for row in range(h):
            for col in range(w):
                if self.new_random_at_each_step:
                    random_list = list(np.random.randn(self.number_of_random_variables))
                # normalise first
                r, c = row / (h-1), col / (w-1)
                # make inputs between -1 and 1
                r = (r - 0.5) * 2
                c = (c - 0.5) * 2
                input = [r, c] + random_list
                output[row, col] = net.activate(input)[0]

        # Threshold        
        thresh = 0.5
        if np.any(output < -0.01): thresh = 0

        return MazeLevel.from_map((output > thresh).astype(np.int32))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(number_of_random_variables={self.number_of_random_variables}, new_random_at_each_step={self.new_random_at_each_step})"


class GenerateMazeLevelsUsingMoreContext(NeatLevelGenerator):
    def __init__(self, game: MazeGame, context_size: int = 1, number_of_random_variables: int = 2,
                  do_padding_randomly: bool = False,
                  random_perturb_size: float = 0):
        super().__init__(number_of_random_variables)
        self.game = game
        self.context_size = context_size
        self.do_padding_randomly = do_padding_randomly
        self.random_perturb_size = random_perturb_size
        self.tile_size = 2 * context_size + 1
        # assert self.tile_size == 5
    
    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        return self.generate_maze_level_using_tiling_bigger_sizes(input, net)

    def generate_maze_level_using_tiling_bigger_sizes(self, input: np.ndarray, net: LevelNeuralNet) -> Level:
        """
        """

        h, w = self.game.level.height, self.game.level.width
        half_tile = self.tile_size // 2
        
        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = 1.0 * (np.random.rand(h + 2 * half_tile, w + 2 * half_tile) > 0.5)
        else:
            output = np.zeros((h + half_tile * 2, w + half_tile * 2)) - 1 # pad it
            output[half_tile:-half_tile, half_tile:-half_tile] = 1.0 * (np.random.rand(h, w) > 0.5) # initial level

        input_list = list(input)
        assert output[half_tile:-half_tile, half_tile:-half_tile].sum() != 0
        output[half_tile:-half_tile, half_tile:-half_tile] = 1*(output[half_tile:-half_tile, half_tile:-half_tile] > 0.5)
        for row in range(half_tile, h + half_tile):
            for col in range(half_tile, w + half_tile):
                # get state
                little_slice = output[row - half_tile: row + half_tile + 1, col - half_tile: col + half_tile + 1]
                # This should be a 3x3 slice now.
                assert little_slice.shape == (self.tile_size, self.tile_size)
                total = self.tile_size * self.tile_size
                little_slice = little_slice.flatten()
                
                # Remove the middle element, which corresponds to the current cell.
                little_slice_list = list(little_slice)
                little_slice_list.pop(total//2)
                assert len(little_slice_list) == total - 1, f"{len(little_slice)} != {total-1}"
                
                # Add in random input.
                little_slice_list.extend(input_list)
                
                input_to_net = little_slice_list
                assert len(input_to_net) == total -1 + self.number_of_random_variables
                if self.random_perturb_size != 0:
                    # Perturb input randomly.
                    input_to_net = np.add(input_to_net, np.random.randn(len(input_to_net)) * self.random_perturb_size)
                
                output_tile = net.activate(input_to_net)[0]
                # Threshold
                output[row, col] = (output_tile > 0.5) * 1.0
        thresh = 0.5
        # if np.any(output < -0.1): thresh = 0
        # Take only relevant parts.
        output = output[half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (h, w)
        return MazeLevel.from_map((output > thresh).astype(np.int32))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size}, context_size={self.context_size})"


if __name__ == "__main__":
    g = GenerateMazeLevelsUsingTiling(MazeGame(MazeLevel()), tile_size=2)
    g.generate_maze_level_using_tiling_bigger_sizes(np.random.randn(2), None)
    pass