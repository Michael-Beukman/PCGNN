#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import sys
from common.utils import get_date
from gym_pcgrl.envs.probs.smb_prob import SMBProblem
import model
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from timeit import default_timer as tmr
import tensorflow as tf
import numpy as np
import os
import pickle
os.environ['WANDB_SILENT'] = 'True'
import wandb


n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 100:
           #pdb.set_trace()
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, we save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
            else:
                print("Saving latest model")
                _locals['self'].save(os.path.join(log_dir, 'latest_model.pkl'))
        else:
            print('{} monitor entries'.format(len(x)))
            pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    np.random.seed(kwargs['seed'])
    tf.random.set_random_seed(kwargs['seed'])

    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    if representation == 'wide':
        policy = FullyConvPolicyBigMap
        if game == "sokoban":
            policy = FullyConvPolicySmallMap
    else:
        policy = CustomPolicyBigMap
        if game == "sokoban":
            policy = CustomPolicySmallMap
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    elif game == 'smb':
        kwargs['cropped_size'] = 114 * 2
    elif game == 'larger_binary' or game == 'large_binary':
        kwargs['cropped_size'] = 40
    
    
    n = max_exp_idx(exp_name)
    global log_dir
    if not resume:
        n = n + 1
    log_dir = 'runs/all_pcgrl_1007/{}_good_{}_{}_{}_seed_{}_date_{}'.format(exp_name, n, 'log', steps, kwargs['seed'], get_date())
    if not resume:
        os.makedirs(log_dir, exist_ok=True)
    else:
        print("LOG DIR  = ", log_dir)
        raise Exception("Cannot resume")
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    used_dir = log_dir
    if not logging:
        used_dir = None
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
    if not resume or model is None:
        model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    else:
        model.set_env(env)
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

################################## MAIN ########################################
experiment = None
assert len(sys.argv) == 4, "Must have 3 parameters"
_, game, representation, seed = sys.argv
seed = int(seed)
steps = 1e8

assert representation in ['turtle', 'wide']
assert game in ['smb', 'binary', 'large_binary']
assert 0 <= seed <= 10

print(f"Running {game} with {representation} for {steps} steps. SEED IS ", seed, 'mode is', representation);
render = False
logging = True
n_cpu = 30
kwargs = {
    'resume': False,
    'seed': seed
}


if __name__ == '__main__':
    F = f'../../../results/all_pcgrl/{game}/{representation}/{steps}/{get_date()}'
    wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True, project="Train_PCGRL", group=F + f"-{n_cpu}")
    dir = F
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, 'data.p'), 'wb+') as f:
        pickle.dump({
            'train_time': 24 * 3600 * 3,
            'log_dir': log_dir,
            'is_at_begin': True,
            'params': {
                'seed': seed,
                'game': game,
                'representation': representation,
            }
        }, f)


    start = tmr()
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
    end = tmr()
    print(f"PCGRL running {game} with repr {representation} for {steps} steps, took {end - start}s")


    with open(os.path.join(dir, 'data.p'), 'wb+') as f:
        pickle.dump({
            'train_time': end - start,
            'log_dir': log_dir,
            'is_at_begin': False,
            'params': {
                'seed': seed,
                'game': game,
                'representation': representation,
            }
        }, f)
