import os
import gym
from stable_baselines import A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results
from stable_baselines.bench import Monitor
import numpy as np
from matplotlib import pyplot as plt

def get_para_dict(gamma, n_step, learning_rate, alpha, epsilon):
    para_dict = {}

    para_dict["gamma"] = gamma
    para_dict["n_step"] = n_step
    para_dict["learning_rate"] = learning_rate
    para_dict["alpha"] = alpha
    para_dict["epsilon"] = epsilon

    return para_dict

def run_episode_mean_reward_experiment(game_name, policy, parameter_dict, timesteps=25000, render=False, atari_game=False):

    # # make parallel environment
    # env = make_vec_env(game_name, n_envs=4)
    
    # make environment
    if not atari_game:
        env = gym.make(game_name)
    else:
        env = make_atari_env(game_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)

    # model defination
    model = A2C(policy, env, verbose=1, gamma=parameter_dict["gamma"]
                                      , n_steps=parameter_dict["n_step"]
                                      , alpha=parameter_dict["alpha"]
                                      , learning_rate=parameter_dict["learning_rate"]
                                      , epsilon=parameter_dict["epsilon"])

    episode_rewards = []
    for i in range(timesteps):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
        episode_rewards.append(reward_sum)
        if i % 100==0:
            print("finished episode " + str(i))
            print(np.mean(episode_rewards))
    
    return np.mean(episode_rewards)
            
def run_reward_curve_experiments(game_name, policy, parameter_dict, timesteps=25000, atari_game=False):
    log_dir = "/tmp/"

    # make environment
    if not atari_game:
        env = gym.make(game_name)
    else:
        env = make_atari_env(game_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
    
    env = Monitor(env, log_dir)

    # model defination
    model = A2C(policy, env, verbose=1, gamma=parameter_dict["gamma"]
                                      , n_steps=parameter_dict["n_step"]
                                      , alpha=parameter_dict["alpha"]
                                      , learning_rate=parameter_dict["learning_rate"]
                                      , epsilon=parameter_dict["epsilon"])
    model.learn(timesteps)

    results_plotter.plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "A2C "+game_name)
    plt.show()




 