import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

render_last = True
episodes = 10000
alpha = 0.9
gamma = 0.95
epsilon = 0.8
reduction = epsilon / episodes
# reduction_alpha = (alpha - 0.1) / episodes
B_velocity = 150
B_position = 100


def discretize (observation, low, high): # np.array([B_position, B_velocity]))
    return ((observation - low) / ((high - low) / np.array([B_position, B_velocity]))) // 1


def main(epsilon=epsilon, alpha=alpha):
    gym.envs.register(
        id='MountainCarMyEasyVersion-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=100000,  # MountainCar-v0 uses 200
    )

    env = gym.make('MountainCarMyEasyVersion-v0')

    # discretize state space
    high = env.observation_space.high
    low = env.observation_space.low

    Q_values = np.random.rand(3, B_position, B_velocity) * 0.001
    timesteps_no = [0] * episodes

    for i in range(episodes):
        timesteps = 0
        done = False
        observation = env.reset()
        d_s = discretize(observation, low, high)

        while not done:
            if (i >= (episodes - 20)) and render_last:
                env.render()

            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q_values[:, int(d_s[0]), int(d_s[1])])
            else:
                action = np.random.randint(0, env.action_space.n)

            observation_new, reward, done, info = env.step(action)
            if done and (observation_new[0] >= 0.5):
                Q_values[action][int(d_s[0])][int(d_s[1])] = reward
            else:
                d_s_new = discretize(observation_new, low, high)
                Q_values[action][int(d_s[0])][int(d_s[1])] += alpha * (reward + \
                                                                       (gamma * np.max(
                                                                           Q_values[:, int(d_s_new[0]), int(d_s_new[1])])) \
                                                                       - Q_values[action][int(d_s[0])][int(d_s[1])])
                d_s = d_s_new
            timesteps += 1

        if i % 100 == 0:
            print(f"Episode {i} finished after ", timesteps, f"timesteps. epsilon:{epsilon}, alpha{alpha}")

        epsilon -= reduction
        # alpha -= reduction_alpha
        timesteps_no[i] = timesteps



    env.close()

    for i in range(3):
        sns.set_palette(sns.color_palette("husl", 15))
        sns.heatmap(Q_values[i][:][:], linewidth=0.0) # , vmin=-10, vmax=4)
        plt.axis('off')
        plt.xlabel("Velocity")
        plt.ylabel("Position")
        plt.title(f"Action {i}")
        plt.show()

    optimal_policy = Q_values.max(axis=0)
    sns.set_palette(sns.color_palette("husl", 15))
    sns.heatmap(optimal_policy, linewidth=0.0) # , vmin=-10, vmax=)
    plt.axis('off')
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title(f"Value function")
    plt.show()

    plt.plot(timesteps_no)
    plt.xlabel("episodes")
    plt.ylabel("timesteps")
    plt.ylim(0, 5000)
    plt.show()


if __name__ == "__main__":
    main()
