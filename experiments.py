import os
import gym
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines.common import make_vec_env
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results
from stable_baselines.bench import Monitor
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np
from matplotlib import pyplot as plt

def get_a2c_para_dict(gamma, n_step, learning_rate, alpha, epsilon):
    para_dict = {}

    para_dict["gamma"] = gamma
    para_dict["n_step"] = n_step
    para_dict["learning_rate"] = learning_rate
    para_dict["alpha"] = alpha
    para_dict["epsilon"] = epsilon

    return para_dict

def get_ddpg_para_dict(gamma, nb_train_steps, nb_rollout_steps, nb_eval_steps, 
                       batch_size, actor_lr, critic_lr, buffer_size, reward_scale):
    para_dict = {}

    para_dict["gamma"] = gamma
    para_dict["nb_train_steps"] = nb_train_steps
    para_dict["nb_rollout_steps"] = nb_rollout_steps
    para_dict["nb_eval_steps"] = nb_eval_steps
    para_dict["batch_size"] = batch_size
    para_dict["actor_lr"] = actor_lr
    para_dict["critic_lr"] = critic_lr
    para_dict["buffer_size"] = buffer_size
    para_dict["reward_scale"] = reward_scale

    return para_dict

def run_episode_mean_reward_experiment_a2c(game_name, solved_score, policy, parameter_dict, episode=500, timesteps=25000, render=False, atari_game=False):
    
    # make environment
    if not atari_game:
        env = make_vec_env(game_name)
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
    total_timesteps = 0
    for i in range(episode):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            total_timesteps += 1
            if total_timesteps > timesteps:
                break

            reward_sum += reward
            if render == True:
                env.render()
        episode_rewards.append(reward_sum)

        if i % 100 == 0:
            print("episode: "+str(i))
        # print('\rEpisode {}, Episode Score: {}, Max: {:.2f}, Min: {:.2f}, Steps: {}'\
        #           .format(i, reward_sum, np.max(episode_rewards), np.min(episode_rewards), total_timesteps), end="\n")

        if reward_sum >= solved_score:
            break
    # print('\rAverage Rewards {}'.format(np.mean(episode_rewards)))
    # plt.plot(np.arange(1, len(episode_rewards)+1), episode_rewards)
    # plt.ylabel('episode reward')
    # plt.xlabel('# of episode')
    # plt.title('a2c '+game_name)
    # plt.show()
    return episode_rewards

            
def run_experiments_moniter_a2c(game_name, policy, parameter_dict, timesteps=25000, atari_game=False):
    log_dir = "/tmp/"

    # make environment
    if not atari_game:
        env = make_vec_env(game_name)
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


def run_episode_mean_reward_experiment_ddpg(game_name, solved_score, policy, parameter_dict, episode=500 ,timesteps=25000, render=False, atari_game=False):

    # make environment
    if not atari_game:
        env = make_vec_env(game_name)
    else:
        env = make_atari_env(game_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # model defination
    model = DDPG(policy, env, action_noise=action_noise ,verbose=1, gamma=parameter_dict["gamma"]
                                        , nb_train_steps=parameter_dict["nb_train_steps"]
                                        , nb_rollout_steps=parameter_dict["nb_rollout_steps"]
                                        , nb_eval_steps=parameter_dict["nb_eval_steps"]
                                        , batch_size=parameter_dict["batch_size"]
                                        , actor_lr=parameter_dict["actor_lr"]
                                        , critic_lr=parameter_dict["critic_lr"]
                                        , buffer_size=parameter_dict["buffer_size"]
                                        , reward_scale=parameter_dict["reward_scale"])

    episode_rewards = []
    total_timesteps = 0
    for i in range(episode):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_timesteps += 1

            if total_timesteps > timesteps:
                break

            if render == True:
                env.render()

            reward_sum += reward
        episode_rewards.append(reward_sum)

        if i % 100 == 0:
            print("episode: "+str(i))
        
        # print('\rEpisode {}, Episode Score: {}, Max: {:.2f}, Min: {:.2f}, total_steps: {:.2f}'\
        #           .format(i, reward_sum, np.max(episode_rewards), np.min(episode_rewards), total_timesteps), end="\n")

        if reward_sum >= solved_score:
            break
    # print('\rAverage Rewards {}'.format(np.mean(episode_rewards)))
    # plt.plot(np.arange(1, len(episode_rewards)+1), episode_rewards)
    # plt.ylabel('episode reward')
    # plt.xlabel('# of episode')
    # plt.title('ddpg '+game_name)
    # plt.show()
    return episode_rewards
 