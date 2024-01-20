# from gymenv import Gym
import gym
from algos import A2C, PPO


# positions = [(-0.5, 0.5)]
# angles = [0, -90]

# env = Gym(disc_action=False, positions=positions, angles=angles)

env = gym.make('BipedalWalker-v3', render_mode = 'rgb_array')


agent = PPO(env=env, k_epochs=10, net_type='actor-critic',
            name='ppo_bipdel', act_space='cont', min_batch_size=2048,
            batch_size=128, lr=0.0001, hid_layer=[64, 64], std_min_clip=0.05, eps_clip=0.2)

epoch = agent.check_status_file()

for ep in range(epoch, 50001):
    done = False
    state = env.reset()[0]
    ep_reward = 0
    steps = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
        agent.buffer.add_experience(state, action, next_state, reward, done)
        state = next_state
        ep_reward += reward
        steps += 1
        if steps >= 500:
            break
    agent.write_plot_data(ep_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_reward)
    print(f'ep. {ep}\tepisode rewards: {ep_reward}')