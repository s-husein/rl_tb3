from gymenv import Gym
from algos import RND_PPO
import numpy as np

positions = [(1, -1), (1, -2), (4, -1), (3, -1), (3, -2), (4, -2), (5, -1), (5, -3), (1, -4),
             (1, -5), (1, -6), (1, -8), (1, -10), (2, -10), (2, -8), (2, -6), (3, -9), (3, -8),
             (3, -7), (4, -8), (5, -7), (4, -10), (5, -10), (3, -4), (4, -4), (3, -5), (4, -5), (4, -6),
             (5, -7), (6, -7), (7, -9), (7, -10), (7, -7)]

angles = np.arange(0, 360, 15)
max_steps = 1000
act_space = 'cont'
hid_layers = [256, 256]
conv_layers = [[16, 3, 1],
               [32, 3, 1],
               [64, 3, 1]]
max_pool = [2, 2]

env = Gym(action_space=act_space, positions=positions, angles=angles, conv_layers=conv_layers, obs_scale_factor=0.1)

agent = RND_PPO(env, k_epochs=10, batch_size=64, hid_layer=hid_layers, conv_layers=conv_layers, min_batch_size=2048,
                actor_lr=0.00003, critic_lr=0.00007, pred_lr=0.0001, act_space=act_space, name='rnd_ppo',
                std_min_clip=0.1, eps_clip=0.3, beta=0.001)

epoch = agent.check_status_file()

for ep in range(epoch, 5001):
    except_flag = False
    done = False
    try:
        state = env.reset()[0]
    except:
        ep -= 1
        continue
    ep_reward = 0
    steps = 0
    while not done:
        action = agent.act(np.expand_dims(state, 0))
        try:
            next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
            env.render()
        except:
            except_flag = True
            break
        agent.buffer.add_experience(state, action, next_state, reward, done)
        state = next_state
        ep_reward += reward
        steps += 1
        if steps >= max_steps:
            break
    if except_flag:
        ep -= 1
        continue
    print(f'ep. {ep}\tepisode rewards: {ep_reward}')
    agent.write_plot_data(ep_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_reward)