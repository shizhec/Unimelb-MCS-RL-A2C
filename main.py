import experiments
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, MlpLstmPolicy, CnnLstmPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_mlpPolicy
from loguniform import LogUniform
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # a2c_para_dict = experiments.get_a2c_para_dict(gamma=0.99, n_step=5, learning_rate=1e-06 , alpha=0.99, epsilon=1e-05)
    # run the a2c problem
    # experiments.run_episode_mean_reward_experiment_a2c('CartPole-v1', 195.0, MlpPolicy, a2c_para_dict, episode=10000, timesteps=1e7, render=False, atari_game=False)
    # experiments.run_experiments_moniter_a2c('CartPole-v1', MlpPolicy, a2c_para_dict)

    # ddpg_para_dict = experiments.get_ddpg_para_dict(gamma=0.99, nb_train_steps=50, nb_rollout_steps=100, 
    #                                                 nb_eval_steps=100, batch_size=128, actor_lr=0.0001, critic_lr=0.001, buffer_size=50000, reward_scale=1.0)
    # experiments.run_episode_mean_reward_experiment_ddpg('LunarLanderContinuous-v2', 200.0, ddpg_mlpPolicy, ddpg_para_dict, episode=10000, timesteps=1e7, render=False, atari_game=False)


    # run_a2c_learning_rate_exp('CartPole-v1', 195, MlpPolicy)
    run_ddpg_buffersize_exp('LunarLanderContinuous-v2', 200.0, ddpg_mlpPolicy)




def run_a2c_learning_rate_exp(game, pass_score, policy):
    max_reward_list = []
    # d1 = LogUniform(0.000001, 0.01)
    # learning_rate_list = d1.rvs(100)
    learning_rate_list = np.arange(0.000001, 0.01, 0.00008)

    i = 1
    for learning_rate in learning_rate_list:
        print('\rrunning experiment {} with learning_rate {}'.format(i, learning_rate))
        a2c_para_dict = experiments.get_a2c_para_dict(gamma=0.99, n_step=5, learning_rate=learning_rate , alpha=0.99, epsilon=1e-05)
        rewards = experiments.run_episode_mean_reward_experiment_a2c(game, pass_score, policy, a2c_para_dict, episode=10000, timesteps=1e7, render=False, atari_game=False)
        max_reward_list.append(np.max(rewards))

        i += 1
    
    
    plt.plot(learning_rate_list, max_reward_list, 'ro')
    plt.xlabel('learning_rate')
    plt.ylabel('max_reward')
    plt.show()

def run_ddpg_buffersize_exp(game, pass_score, policy):
    buffersize_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
    max_reward_list = []
    mean_reward_list = []

    i = 1
    for buffersize in buffersize_list:
        print('\rrunning experiment {} with buffersize {}'.format(i, buffersize))
        ddpg_para_dict = experiments.get_ddpg_para_dict(gamma=0.99, nb_train_steps=50, nb_rollout_steps=100, 
                                        nb_eval_steps=100, batch_size=128, actor_lr=0.0001, critic_lr=0.001, buffer_size=buffersize, reward_scale=1.0)
        rewards = experiments.run_episode_mean_reward_experiment_ddpg(game, pass_score, policy, ddpg_para_dict, 
                                                                episode=10000, timesteps=1e7, render=False, atari_game=False)
        max_reward_list.append(np.max(rewards))
        mean_reward_list.append(np.mean(rewards))
        i += 1

    plt.plot(buffersize_list, mean_reward_list)
    plt.xlabel('buffersize')
    plt.ylabel('mean_reward')
    plt.show()


if __name__ == "__main__":
    main()
