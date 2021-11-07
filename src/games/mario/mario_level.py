from typing import Dict, List, Union
from matplotlib import pyplot as plt

import numpy as np
from common.types import TileMap
from games.level import Level
from PIL import Image
import os
from gym_pcgrl.envs.probs.smb.engine import State as MyState
from gym_pcgrl.envs.probs.smb.engine import Node as MyNode
MarioState = MyState
MarioNode = MyNode
class MarioLevel(Level):
    """A single level of a Mario Game. There are different tile types, like 'empty', 'solid', 'enemy', etc.
    """
    def __init__(self, width: int = 114, height: int = 14):
        tiles = ["empty", "solid", "enemy", "brick", "question", "coin", "tube"]
        types = {i: v for i, v in enumerate(tiles)}
        # Initial dummy level, just a solid floor.
        map = np.zeros((height, width), dtype=np.int32)
        # Make the floor solid
        map[-1, :] = 1
        self._graphics = None
        self._tile_size = 16
        self._border_tile = tiles[0]
        self._border_size = (0, 0)
        super().__init__(width, height, tile_types=types, map=map)
    
    def get_string_map(self) -> List[List[str]]:
        map = self.map
        li = np.empty(map.shape, dtype=object)
        for y in range(len(map)):
            for x in range(len(map[y])):
                li[y][x] = self.tile_types[map[y][x]]
        return li

    def show(self, do_show: bool = True) -> None:
        # Lots of this is from PCGRL https://github.com/amidos2006/gym-pcgrl
        map = self.map
        li = self.get_string_map()
        map = li
        new_map = []
        for y in range(len(map)):
            new_map.append([])
            for x in range(3):
                if y < self.height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")
            for x in range(len(map[y])):
                value = map[y][x]
                if map[y][x] == "solid" and y < self.height - 2:
                    value = "solid_above"
                if map[y][x] == "tube":
                    if y >= 1 and map[y-1][x] != "tube":
                        value = "top"
                    if x >= 1 and map[y][x-1] != "tube":
                        value += "_left"
                    else:
                        value += "_right"
                new_map[y].append(value)
            for x in range(3):
                if y < self.height - 2:
                    new_map[y].append("empty")
                else:
                    new_map[y].append("solid")

        new_map[-3][1] = "player"
        new_map[-3][-2] = "solid_above"
        for y in range(3, len(map) - 3):
            new_map[y][-2] = "pole"
        new_map[1][-2] = "pole_top"
        new_map[2][-2] = "pole_flag"
        new_map[2][-3] = "flag"

        
        
        map = new_map

        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/assets/empty.png").convert('RGBA'),
                # "solid": Image.open(os.path.dirname(__file__) + "/assets/solid_floor.png").convert('RGBA'),
                "solid_above": Image.open(os.path.dirname(__file__) + "/assets/solid_air.png").convert('RGBA'),
                # Michael changed this to get a more uniform level aesthetic
                "solid": Image.open(os.path.dirname(__file__) + "/assets/solid_air.png").convert('RGBA'),
                "enemy": Image.open(os.path.dirname(__file__) + "/assets/enemy.png").convert('RGBA'),
                "brick": Image.open(os.path.dirname(__file__) + "/assets/brick.png").convert('RGBA'),
                "question": Image.open(os.path.dirname(__file__) + "/assets/question.png").convert('RGBA'),
                "coin": Image.open(os.path.dirname(__file__) + "/assets/coin.png").convert('RGBA'),
                "top_left": Image.open(os.path.dirname(__file__) + "/assets/top_left.png").convert('RGBA'),
                "top_right": Image.open(os.path.dirname(__file__) + "/assets/top_right.png").convert('RGBA'),
                "tube_left": Image.open(os.path.dirname(__file__) + "/assets/tube_left.png").convert('RGBA'),
                "tube_right": Image.open(os.path.dirname(__file__) + "/assets/tube_right.png").convert('RGBA'),
                "pole_top": Image.open(os.path.dirname(__file__) + "/assets/poletop.png").convert('RGBA'),
                "pole": Image.open(os.path.dirname(__file__) + "/assets/pole.png").convert('RGBA'),
                "pole_flag": Image.open(os.path.dirname(__file__) + "/assets/flag.png").convert('RGBA'),
                "flag": Image.open(os.path.dirname(__file__) + "/assets/flagside.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/assets/player.png").convert('RGBA')
            }
        self._border_size = (0, 0)
        
        self._border_size = (3, 0)


        tiles = self.tile_types

        full_width = len(map[0])+2*self._border_size[0]
        full_height = len(map)+2*self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,255))
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))
        for y in range(len(map)):
            for x in range(len(map[y])):
                lvl_image.paste(self._graphics[map[y][x]], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size))
        if do_show:
            plt.figure(figsize=(20,5))
            plt.imshow(lvl_image)
        return lvl_image
    

    def string_representation_of_level(self, show_enemies: bool = False) -> str:
        """Returns a string representation of the level. From the PCGRL codebase
        
            show_enemies:
                If this is true, then we show enemies as an x. Otherwise they are ignored.
        Returns:
            str
        """
        def single_level(level: Level):
            gameCharacters=" # ## #"
            if show_enemies:
                gameCharacters=" #x## #"
            string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(level.tile_types.values()))
            lvlString = ""
            height, width = level.height, level.width
            map = level.get_string_map()
            for i in range(len(map)):
                if i < height - 3:
                    lvlString += "   "
                elif i == height - 3:
                    lvlString += " @ "
                else:
                    lvlString += "###"
                for j in range(len(map[i])):
                    string = map[i][j]
                    lvlString += string_to_char[string]
                if i < height - 3:
                    lvlString += " | "
                elif i == height - 3:
                    lvlString += " # "
                else:
                    lvlString += "###"
                lvlString += "\n"

            return lvlString

        return single_level(self)
    @staticmethod
    def from_map(map: np.ndarray) -> "Level":
        level = MarioLevel(map.shape[1], map.shape[0])
        level.map = map
        return level
    
    def to_mario_ai_string_format(self) -> str:
        """This makes a string representation of the level as in the Mario-AI-Framework.

        Returns:
            str: [description]
        """
        []
        tile_mapping = {
            self.tile_types_reversed[k]: v for k, v in {
                "empty": '-', 
                "solid": 'X', 
                "enemy": 'g', 
                "brick": '#', 
                "question": "Q", 
                "coin": 'o', 
                "tube": 't'
            }.items()
        }
        

        # add in flag and Mario
        S = ""
        def actual_tile(x, y):
            if x < 0 or x >= self.map.shape[1]:
                return '-'
            return str(tile_mapping[self.map[y, x]])
        for y in range(self.map.shape[0]):
            for x in range(-3, self.map.shape[1] + 3):
                if x < 0:
                    if y >= self.map.shape[0] - 2:
                        S += "X"
                    elif y == self.map.shape[0] - 3 and x == -2:
                        S += "M"
                    else:
                        S += actual_tile(x, y)
                elif x >= self.map.shape[1]:
                    if y >= self.map.shape[0] - 1:
                        S += "X"
                    elif y >= self.map.shape[0] - 2 and x == self.map.shape[1] + 1:
                        S += "X"
                    elif y >= self.map.shape[0] - 3 and x == self.map.shape[1] + 1:
                        S += "F"
                    else:
                        S += actual_tile(x, y)
                else:
                    S += actual_tile(x, y)     
            S += "\n"
        return S[:-1]

    def to_file_for_java(self, filename: str):
        """Writes the java string repr to filename

        Args:
            filename (str): [description]
        """
        with open(filename, 'w+') as f:
            f.write(self.to_mario_ai_string_format())
    
    def to_file(self, filename: str):
        self.to_file_for_java(filename)

if __name__ == '__main__':
    level = MarioLevel()
    print(level.to_mario_ai_string_format());
    exit()
    # level.show()
    
    gameCharacters=" # ## #"
    gameCharacters="1234567"
    string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(level.tile_types.values()))
    print(string_to_char)
    print(level.tile_types.values())
    pass

    
