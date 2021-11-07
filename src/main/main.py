import os
import fire
from matplotlib import pyplot as plt
from games.mario.java_runner import run_java_task
from games.mario.mario_level import MarioLevel

from novelty_neat_generate import generate_level


def main_func(
        method: str = 'noveltyneat',
        game: str = 'mario',
        command: str = 'generate',
        width: int = -1,
        height: int = -1,
        filename: str = 'test_level.txt'
):
    """This runs some commands 

    Args:
        method (str): One of ['noveltyneat']. This is the method that will be used. # TODO: , 'directga', 'pcgrl'
        game (str): One of ['maze', 'mario']. This is the method that will be used.
        command (str): One of ['generate', 'human-play', 'agent-play']
    """
    print(f"You chose {method} and {command}")

    if width < 0:
        width = 14 if game.lower() == 'maze' else 114
    if height  < 0:
        height = width if game.lower() == 'maze' else 114

    assert width >= 5 and height >= 5, "Please make width and height at least 5"

    d = 'main/levels/'; 
    where_to_save_load = os.path.join(d, f'{game}_{filename}')
    if method == 'noveltyneat' and command == 'generate':
        level = generate_level(game, width, height)
        print(level)
        level.show()
        plt.show()
        os.makedirs(d, exist_ok=True)
        level.to_file(where_to_save_load)
        print(f"Saved level to {where_to_save_load}!")
    
    if 'play-' == command[:5]:
        if not os.path.exists(where_to_save_load):
            print(f"{where_to_save_load} does not exist. Please try again")
            exit(1)
    if command == 'play-human':
        print("Controls:")
        print("\tArrows to Move")
        print("\ts to jump")
        assert game == 'mario', "Can only play Mario levels"
        run_java_task(["Human_Play", where_to_save_load, "0", "0", "false"])
    elif command == 'play-agent':
        assert game == 'mario', "Can only play Mario levels"
        run_java_task(["Visual_Test", where_to_save_load, "200", "1", "false"])



if __name__ == '__main__':
    fire.Fire(main_func)