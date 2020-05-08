import experiments
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, BasePolicy, FeedForwardPolicy, ActorCriticPolicy
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    para_dict = experiments.get_para_dict(gamma=0.99, n_step=5, learning_rate=0.0007, alpha=0.99, epsilon=1e-05)
    # episode_mean_rewards = experiments.run_episode_mean_reward_experiment("MountainCarContinuous-v0", MlpPolicy, para_dict, timesteps=25000)
    # print(episode_mean_rewards)



    experiments.run_reward_curve_experiments("Acrobot-v1", MlpPolicy, para_dict, timesteps=25000)








if __name__ == "__main__":
    main()

# 22.16092
# 22.20868