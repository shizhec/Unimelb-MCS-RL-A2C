import gym
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, BasePolicy, FeedForwardPolicy, ActorCriticPolicy
from stable_baselines import A2C
from stable_baselines.common import make_vec_env


def run_experiment(game_name, parameter_dict, render=False):
    reward_list = []

    # parallel environment
    env = make_vec_env(game_name, n_envs=parameter_dict["n_envs"])

    # model defination
    model = A2C(parameter_dict["policy"], env, verbose=1)
    model.load_parameters(parameter_dict)

    model.learn(total_timesteps=parameter_dict["total_timesteps"])
    model.save("a2c_"+ game_name)
    
    del model

    model.load("a2c_"+ game_name)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if render==True:
            env.render(mode='rgb_array')
        reward_list.append(rewards)
            





 