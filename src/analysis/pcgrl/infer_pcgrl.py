
import os
import pickle

import numpy as np
from games.mario.mario_level import MarioLevel
import model
from stable_baselines import PPO2
import time
from utils import make_vec_envs
import matplotlib.pyplot as plt
from games.maze.maze_level import MazeLevel
from novelty_neat.maze.utils import path_length
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from timeit import default_timer as timer

def infer(representation = 'wide', 
          game='binary',
          seed=1, name = 'None',
          model_path=None,
          where_to_save=None, train_time=None):

    run_no = seed if (seed <= 3 or representation == "turtle") else seed + 1
    if representation == 'wide' and game == 'smb' and seed >= 2:
        run_no = seed + 2
    
    if model_path is None:
        model_path = f'external/gym-pcgrl/runs/{game}_{representation}_{run_no}_log_100000000.0_seed_{seed}/best_model.pkl'

    kwargs = {
        'change_percentage': 1,
        'verbose': True
    }
    
    if game == "binary" or game == 'large_binary':
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap

    agent = PPO2.load(model_path)

    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    elif game == 'smb':
        kwargs['cropped_size'] = 114 * 2
    elif game == 'large_binary':
        kwargs['cropped_size'] = 40
    

    print("IN inferring mode")
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)

    import copy
    def get_path_length(map):
        level = MazeLevel.from_map(map)
        return path_length(level)

    def plot_map(map, ax, title):
        map = copy.deepcopy(map)
        if map is None: 
            ax.set_title(title)
            return
        if len(map.shape) == 4:
            map = map[0, :, :, 0]

        ax.imshow(1-map, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{title}. Path length = {get_path_length(map)}")

    levels = []

    def get_level_from_obs(obs):
        assert len(obs.shape) == 2
        if game == 'binary':
            return MazeLevel.from_map(obs)
        else:
            return MarioLevel.from_map(obs)

    def my_infer(env, agent, **kwargs):
        plots = False
        obs = env.reset()
        dones = False
        total_rewards = 0
        old_obs = obs
        K = 0
        N = 10
        agent.set_env(env)
        print("SETTED")
        while not dones:
            K += 1
            action, _ = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)

            if 'terminal_observation' in info[0] or K >= N:
                my_obs = info[0].get('terminal_observation', obs)
                if K >= N:
                    print("My terminal obs = ", my_obs.shape)
                    if my_obs.shape[0] == 1 and len(my_obs.shape) == 4:
                        my_obs = my_obs[0]
                if representation == 'turtle':
                    pos = info[0]['original_observation']['pos']
                    map = info[0]['original_observation']['map']
                    x, y = pos
                    h, w = map.shape[:2]
                    h //= 2
                    w //= 2
                    my_obs = map[h - y: h+h - y, w - x: w + w - x]
                    if game == 'smb':
                        my_obs = my_obs[:14]
                if game == 'binary' or game == 'large_binary':
                    if len(my_obs.shape) == 3:
                        my_obs = my_obs[:, :, 0]
                else:
                    my_obs = np.argmax(my_obs, axis=-1)
                levels.append(get_level_from_obs(my_obs))
            else:
                my_obs = obs[0, :, :]
            total_rewards += rewards
            old_obs = obs
            if dones or K >= N:
                print(f"Stopped After {K} steps, and dones = {dones}.{'BAD' if K >= N else ''}")
                break
    start_time = timer()
    for i in range(100):
        print(f"\n{i}/100")
        my_infer(env, agent, **kwargs)
    end_time = timer()

    assert len(levels) == 100, "We must have 100 levels"
    total_time_per_level = (end_time - start_time) / len(levels)
    proper_levels = levels

    good_name = 'maze' if game == 'binary' else 'mario'
    if where_to_save is None:
        dir = f'results/{good_name}/pcgrl/{representation}/{name}/'
    else:
        dir = where_to_save
    os.makedirs(dir, exist_ok=True)
    print("TIME PER LEVEL ", total_time_per_level)
    with open(os.path.join(dir, f'run_seed_{seed}.p'), 'wb+') as f: 
        pickle.dump({'levels': proper_levels, 'path': model_path, 'time_per_level': total_time_per_level, 'train_time': train_time}, f)
    return levels



if __name__ == '__main__':
    def all_pcgrl_turtle_binary():
        files = [
            '../results/all_pcgrl/binary/turtle/100000000.0/2021-10-07_17-40-17/data.p',
            '../results/all_pcgrl/binary/turtle/100000000.0/2021-10-08_05-54-04/data.p',
            '../results/all_pcgrl/binary/turtle/100000000.0/2021-10-08_05-55-48/data.p',
            '../results/all_pcgrl/binary/turtle/100000000.0/2021-10-08_14-17-14/data.p',
            '../results/all_pcgrl/binary/turtle/100000000.0/2021-10-08_14-17-41/data.p',
        ]
        dir_to_save = '../results/all_pcgrl/binary/turtle/100000000.0/inferred_levels_v2'
        for f in files:
            with open(f, 'rb') as pickle_f:
                D = pickle.load(pickle_f)
                model_path = os.path.join('external/gym-pcgrl', D['log_dir'], 'best_model.pkl')
                seed = D['params']['seed']
            infer('turtle', 'binary', seed, 'pcgrl_all', model_path=model_path, where_to_save=dir_to_save, train_time=D['train_time'])

    def all_pcgrl_wide_binary():
        files = [
            '../results/all_pcgrl/binary/wide/100000000.0/2021-10-10_07-53-22/data.p',
            '../results/all_pcgrl/binary/wide/100000000.0/2021-10-10_07-54-09/data.p',
            '../results/all_pcgrl/binary/wide/100000000.0/2021-10-10_07-54-02/data.p',
            '../results/all_pcgrl/binary/wide/100000000.0/2021-10-10_07-53-36/data.p',
            '../results/all_pcgrl/binary/wide/100000000.0/2021-10-10_07-54-28/data.p',
        ]
        dir_to_save = '../results/all_pcgrl/binary/wide/100000000.0/inferred_levels_v2'
        for f in files:
            with open(f, 'rb') as pickle_f:
                D = pickle.load(pickle_f)
                model_path = os.path.join('external/gym-pcgrl', D['log_dir'], 'best_model.pkl')
                seed = D['params']['seed']
            infer('wide', 'binary', seed, 'pcgrl_all', model_path=model_path, where_to_save=dir_to_save, train_time=D['train_time'])

    HEAD = 'external/gym-pcgrl/runs/all_pcgrl_1007'
    # HEAD = '/tmp//NAME//all_pcgrl_1007'
    def all_pcgrl_turtle_smb():
        print("DOING TURTLE")
        files = [
            'smb_turtle_good_10_log_100000000.0_seed_3_date_2021-10-10_07-53-45',
            'smb_turtle_good_10_log_100000000.0_seed_4_date_2021-10-10_07-54-43',
            'smb_turtle_good_10_log_100000000.0_seed_2_date_2021-10-10_07-54-49',
            'smb_turtle_good_10_log_100000000.0_seed_1_date_2021-10-10_07-54-24',
            'smb_turtle_good_10_log_100000000.0_seed_5_date_2021-10-10_07-54-40',
        ]
        dir_to_save = '../results/all_pcgrl/smb/turtle/100000000.0/inferred_levels_v2'
        for f in files:
            model_path = os.path.join(HEAD, f, 'best_model.pkl')
            seed = int(f.split('seed_')[1].split("_")[0])
            infer('turtle', 'smb', seed, 'pcgrl_all', model_path=model_path, where_to_save=dir_to_save, train_time=259200)
    
    def all_pcgrl_wide_smb():
        print("DOING WIDE")
        files = [
            'smb_wide_good_10_log_100000000.0_seed_4_date_2021-10-10_07-54-05',
            'smb_wide_good_10_log_100000000.0_seed_2_date_2021-10-10_07-55-09',
            'smb_wide_good_10_log_100000000.0_seed_1_date_2021-10-10_07-53-55',
            'smb_wide_good_10_log_100000000.0_seed_5_date_2021-10-10_07-54-16',
            'smb_wide_good_10_log_100000000.0_seed_3_date_2021-10-10_07-55-11',
        ]
        dir_to_save = '../results/all_pcgrl/smb/wide/100000000.0/inferred_levels'
        for f in files:
            model_path = os.path.join(HEAD, f, 'best_model.pkl')
            seed = int(f.split('seed_')[1].split("_")[0])
            infer('wide', 'smb', seed, 'pcgrl_all', model_path=model_path, where_to_save=dir_to_save, train_time=259200)
    

    all_pcgrl_wide_binary()    
    all_pcgrl_turtle_binary()    
    all_pcgrl_turtle_smb() 
    all_pcgrl_wide_smb()