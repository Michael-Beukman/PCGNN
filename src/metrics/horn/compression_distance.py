from pydoc import plain
from typing import List
from matplotlib import pyplot as plt

import numpy as np
from baselines.ga.multi_pop_ga.multi_population_mario import MarioGAPCG
from games.game import Game
from games.level import Level
from games.mario.mario_game import MarioGame
from games.mario.mario_level import MarioLevel
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from metrics.metric import Metric
import gzip
import zlib
class CompressionDistanceMetric(Metric):
    """
        Compression distance metric from Horn et al. (2014)

        Described for platformer games here: N. Shaker, M. Nicolau, G. N. Yannakakis, J. Togelius, and M. O’Neill. Evolving levels for Super Mario Bros using grammatical evolution. In Computational Intelligence and Games (CIG), 2012 IEEE Conference on, pages 304–311. IEEE, 2012.
        And from here: M. Li, X. Chen, X. Li, B. Ma, and P. Vitanyi, “The similarity metric,” ´IEEE Transactions on Information Theory, vol. 50, no. 12, pp. 3250–3264, 2004

        Basically, NCD(x, y) = (C(xy) - min{C(x), C(y)}) / max{C(x), C(y)}
        where C(z) is the compressed size of z
    Args:
        Metric ([type]): [description]
    """
    def __init__(self, game: Game, 
                use_combined_features: bool = False,
                do_mario_flat: bool = False) -> None:
        """

        Args:
            game (Game): 
            use_combined_features (bool, optional): Only applicable to Mario. If this is true, use combination of features.
            From here
                Shaker, N., Nicolau, M., Yannakakis, G. N., Togelius, J., & O'neill, M. (2012, September). Evolving levels for super mario bros using grammatical evolution. In 2012 IEEE Conference on Computational Intelligence and Games (CIG) (pp. 304-311). IEEE.
            We consider the start, end (or nothing) of a gap (G)
            We consider the increase, decrease (or staying the same) of the platform height (I)
            The existence of enemies and blocks (0 = none, 1 = enemy, 2 = block, 3 = both) (ED)
            
            And then combine the above to get a number for each index ED + 4 * I + 12 * G, which is then converted into an alphabet letter.


            If this is False, we don't do that, and instead make two strings, one that has the platform height and the other 
            that has the ED string as described above and return the concatenation.

            Defaults to False.

            do_mario_flat (bool, optional). Also only applicable to Mario. If this is true, we use the flattened level of integers as the string to compress. 
                Defaults to False
        """
        super().__init__(game)
        self.use_better_features = use_combined_features
        self.do_mario_flat = do_mario_flat

    def name(self):
        if (isinstance(self.game, MarioGame)) and self.do_mario_flat:
            return "CompressionDistanceMetric_FlatMario"

        if (isinstance(self.game, MarioGame)) and self.use_better_features:
            return "CompressionDistanceMetric_CombinedFeatures"
        return super().name()
    def evaluate(self, levels: List[Level]) -> List[float]:
        if (isinstance(self.game, MarioGame)):
            return self.evaluate_mario(levels)
        
        pairwise_values = []
        string_levels = [
            ''.join(map(str, level.map.flatten())) for level in levels
        ]
        def C(z):
            "Compression length of one string"
            return len(gzip.compress(bytes(z,'utf-8')))
        for i in range(len(string_levels)):
            for j in range(len(string_levels))[i+1:]:
                x = string_levels[i]
                y = string_levels[j]
                xy = x + y
                value = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
                pairwise_values.append(value)

        return pairwise_values

    def evaluate_mario(self, levels: List[Level]) -> List[float]:
        """Evaluates the compression distance for Mario levels by making features from:
            Shaker et al., Feature analysis for modeling game content quality, 2011

            As said in Shaker et al. 2012 Evolving Levels for Super Mario Bros Using Grammatical Evolution,
            we must convert the levels to string representations


            This representation is P + ED, where P is the platform string and ED is the enemy / decoration string,
            so P = ''.join(height_of_platform(i) for i in range(W)])
            so ED = ''.join(has_enemy(i) * 1 + has_block(i) * 2 for i in range(W)])

        Args:
            levels (List[Level]):

        Returns:
            List[float]:
        """
        pairwise_values = []
        def to_string(level: Level):
            platform_string = ''
            enemy_and_decoration_string = ''
            map = level.map
            W, H = level.width, level.height
            for w in range(W):
                column = map[:, w]
                # Platform
                height = np.argwhere(column == level.tile_types_reversed['solid'])
                if len(height) == 0: 
                    P = 0
                else:
                    P = H - height[0]
                platform_string += str(P)

                # decoration
                has_enemy = (column == level.tile_types_reversed['solid']).sum()

                has_decoration = 0
                for name in ["brick", "question", "coin", "tube"]:
                    has_decoration += (column == level.tile_types_reversed[name]).sum()
                has_decoration = 1 if has_decoration > 0 else 0

                num_enemy_dec = has_enemy + has_decoration * 2
                enemy_and_decoration_string += str(num_enemy_dec)
            return platform_string + enemy_and_decoration_string

            pass
        
        def to_string_better(level: Level):
            incdec = ''
            gap    = ''
            enemy_and_decoration_string = ''
            map = level.map
            W, H = level.width, level.height
            prev_height = 0
            for w in range(W):
                column = map[:, w]
                # Platform
                height = np.argwhere(column == level.tile_types_reversed['solid'])

                
                if len(height) == 0: 
                    P = 0
                else:
                    P = H - height[0]
                
                if incdec == '' or P == prev_height:
                    incdec += '0'
                elif P > prev_height:
                    incdec += '1'
                elif P < prev_height:
                    incdec += '2'
                
                if prev_height == 0 and P == 0:
                    # no start / end of gap
                    gap += '0'
                elif prev_height == 0:
                    # prev is 0 but now isn't -> end of gap
                    gap += '1'
                elif P == 0:
                    # Now is 0 but prev isn't -> start of gap
                    gap += '2'
                else:
                    gap += '0'
                # decoration
                prev_height = P
                has_enemy = 1 if (column == level.tile_types_reversed['solid']).sum() > 0 else 0

                has_decoration = 0
                for name in ["brick", "question", "coin", "tube"]:
                    has_decoration += (column == level.tile_types_reversed[name]).sum()
                has_decoration = 1 if has_decoration > 0 else 0

                num_enemy_dec = has_enemy + has_decoration * 2
                enemy_and_decoration_string += str(num_enemy_dec)
            
            new_good_string = ''

            assert len(enemy_and_decoration_string) == len(gap) == len(incdec)
            for index in range(len(gap)):
                g = int(gap[index])
                i = int(incdec[index])
                ed = int(enemy_and_decoration_string[index])

                # convert into one number
                new_one = ed + 4 * (i + 3 * g)
                assert 0 <= new_one <= 35
                if 0 <= new_one < 26:
                    new_good_string += chr(new_one + 97)
                else:
                    new_good_string += chr(new_one - 26 + 65)
            return new_good_string

        def string_flatten(level: Level):
            return ''.join(map(str, level.map.flatten()))

        str_func = to_string if not self.use_better_features else to_string_better
        if self.do_mario_flat:
            str_func = string_flatten

        string_levels = [
            str_func(level) for level in levels
        ]
        def C(z):
            "Compression length of one string"
            return len(gzip.compress(bytes(z,'utf-8')))
        for i in range(len(string_levels)):
            for j in range(len(string_levels))[i+1:]:
                x = string_levels[i]
                y = string_levels[j]
                xy = x + y
                value = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
                pairwise_values.append(value)

        return pairwise_values
    

def test_cd():
    np.random.seed(42)
    levels = [
        MazeLevel.random(),
        MazeLevel.random(),
        MazeLevel(),
        MazeLevel(),
    ]

    cd = CompressionDistanceMetric(MazeGame(MazeLevel()))
    E = cd.evaluate(levels)

    mat = np.zeros((4, 4))
    C = 0
    for i in range(len(levels)):
        temp = [levels[i], levels[i]]
        L = cd.evaluate(temp)
        assert len(L) == 1
        mat[i, i] = L[0]
        for j in range(i+1, len(levels)):
            mat[i, j] = E[C]
            mat[j, i] = E[C]
            C += 1
    
    print(np.round(mat, 2))


if __name__ == '__main__':
    test_cd(); exit()