from typing import List
import numpy as np
from games.game import Game
from games.level import Level
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.types import LevelNeuralNet


def _one_hot(value: int, total: int) -> List[float]:
    arr = [0] * total
    arr[int(value)] = 1.0
    return arr


class GenerateGeneralLevelUsingTiling(NeatLevelGenerator):
    def __init__(self, game: Game,
                 context_size: int = 1, number_of_random_variables: int = 2,
                 do_padding_randomly: bool = False,
                 random_perturb_size: float = 0,
                 predict_size: int = 1,
                 reversed_direction: int = 0,
                 use_one_hot_encoding: int = False
                 ):
        """Generates levels using a tiling approach, i.e. moving through all of the cells, giving the network the surrounding ones and taking the prediction as the current tile.
        Args:
            game (Game): The game to generate for
            context_size (int, optional): How many tiles should the network take in as input to predict the current tile. 
                If this is 1, then the network takes in the surrounding 8 tiles (i.e. x +- 1 and y +- 1). 
                If this is two, then it takes in the surrounding 24 tiles.
                Defaults to 1.

            number_of_random_variables (int, optional): The number of random variables to add to the network. Defaults to 2.
            do_padding_randomly (bool, optional): If this is true, then we don't pad with -1s around the borders, but we instead make those random as well. Defaults to False.
            random_perturb_size (float, optional): If this is nonzero, all inputs to the net, including coordinates and surrounding tiles will be randomly perturbed by a gaussian (mean 0, variance 1) 
                multiplied by this value. Defaults to 0.
            predict_size (int, optional): If this is 1, then the network predicts one tile. If this is two, the network predicts 4 tiles (a 2x2 area), etc.
            reversed_direction (bool, optional): If this is 0, we iterate from the top left to the bottom right. Otherwise, if it is 1, we iterate from the bottom right to the top left.
            If it is 2, we choose random directions each time.
            use_one_hot_encoding (bool, optional). If this is true, then we use one hot encoding for the inputs instead of just ints. Defaults to False
        """
        super().__init__(number_of_random_variables)
        self.game = game
        self.context_size = context_size
        self.do_padding_randomly = do_padding_randomly
        self.random_perturb_size = random_perturb_size
        self.tile_size = 2 * context_size + 1
        self.number_of_tile_types = len(self.game.level.tile_types)
        self.predict_size = predict_size
        self.reversed_direction = reversed_direction
        self.use_one_hot_encoding = use_one_hot_encoding

    def generate_level(self, net: LevelNeuralNet, input: np.ndarray) -> Level:
        h, w = self.game.level.height, self.game.level.width
        half_tile = self.tile_size // 2

        if self.do_padding_randomly:
            # Pad randomly, and don't make the edges special.
            output = np.random.randint(0, self.number_of_tile_types, size=(
                h + 2 * half_tile, w + 2 * half_tile))
        else:
            # pad it with negatives
            output = np.zeros((h + half_tile * 2, w + half_tile * 2)) - 1
            output[half_tile:-half_tile, half_tile:-
                   half_tile] = np.random.randint(0, self.number_of_tile_types, size=(h, w))

        input_list = list(input)
        assert output[half_tile:-half_tile, half_tile:-half_tile].sum() != 0
        output[half_tile:-half_tile, half_tile:-half_tile] = 1 * \
            (output[half_tile:-half_tile, half_tile:-half_tile] > 0.5)
        X = self.predict_size - 1
        num_preds = self.predict_size ** 2

        range_rows = range(half_tile, h + half_tile - X)
        range_cols = range(half_tile, w + half_tile - X)
        if self.reversed_direction == 1:
            range_rows = reversed(range_rows)
            range_cols = reversed(range_cols)
        elif self.reversed_direction == 2:
            if np.random.rand() < 0.5:
                range_rows = reversed(range_rows)
            if np.random.rand() < 0.5:
                range_cols = reversed(range_cols)

        # This is super important, as the reversed thing is a one use iterator!
        # You cannot iterate multiple times!!!!!!!!!!
        range_rows = list(range_rows)
        range_cols = list(range_cols)

        for row in range_rows:
            for col in range_cols:
                # get state
                # Suppose (row, col) is the top left corner of our prediction tile. Then we need to move predict_size - 1 more to the right and down.

                little_slice = output[row - half_tile: row + half_tile +
                                      1 + X, col - half_tile: col + half_tile + 1 + X]
                # This should be a nxn slice now.
                assert little_slice.shape == (
                    self.tile_size + X, self.tile_size + X)
                total = self.tile_size * self.tile_size
                little_slice = little_slice.flatten()

                little_slice_list = list(little_slice)
                if self.predict_size == 1:  # Don't remove the middle tiles if pred size > 1
                    # Remove the middle element, which corresponds to the current cell.
                    little_slice_list.pop(total//2)
                    assert len(little_slice_list) == total - \
                        1, f"{len(little_slice)} != {total-1}"

                if self.use_one_hot_encoding:
                    # now we need to encode the array into a one hot thing.
                    curr_ans = []
                    for value in little_slice_list:
                        curr_ans = curr_ans + _one_hot(value, self.number_of_tile_types)
                    curr_ans = curr_ans
                    little_slice_list = curr_ans
                # Add in random input.
                little_slice_list.extend(input_list)

                input_to_net = little_slice_list

                if self.random_perturb_size != 0:
                    # Perturb input randomly.
                    input_to_net = np.add(input_to_net, np.random.randn(
                        len(input_to_net)) * self.random_perturb_size)

                # Should be a result of size self.number_of_tile_types, so choose argmax.
                output_results = net.activate(input_to_net)
                assert len(output_results) == num_preds * \
                    self.number_of_tile_types, "Network should output the same amount of numbers as there are tile types"
                output_results = np.array(output_results).reshape(
                    (self.number_of_tile_types, self.predict_size, self.predict_size))

                tile_to_choose = np.argmax(output_results, axis=0)
                assert tile_to_choose.shape == (
                    self.predict_size, self.predict_size)
                for i in range(self.predict_size):
                    for j in range(self.predict_size):
                        output[row + i, col + j] = tile_to_choose[i, j]

        output = output[half_tile:-half_tile, half_tile:-half_tile]
        assert output.shape == (h, w)
        return self.game.level.from_map(output)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tile_size={self.tile_size}, number_of_random_variables={self.number_of_random_variables}, do_padding_randomly={self.do_padding_randomly}, random_perturb_size={self.random_perturb_size}, context_size={self.context_size}, number_of_tiles={self.number_of_tile_types}, game={self.game}, predict_size={self.predict_size}, reverse={self.reversed_direction}, use_one_hot_encoding={self.use_one_hot_encoding})"
