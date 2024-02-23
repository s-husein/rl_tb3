from gymenv import Gym
from algos import RND_PPO
import numpy as np
import torch

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
pre_steps = 100

env = Gym(action_space=act_space, positions=positions, angles=angles, conv_layers=conv_layers, obs_scale_factor=0.1)

agent = RND_PPO(env, k_epochs=10, batch_size=32, hid_layer=hid_layers, conv_layers=conv_layers, min_batch_size=2048,
                actor_lr=0.00003, critic_lr=0.00007, pred_lr=0.0001, act_space=act_space, name='rnd_ppo',
                std_min_clip=0.1, eps_clip=0.3, beta=0.001, max_pool=max_pool, act_fn='elu')

epoch = agent.check_status_file()


#taking random steps to initialize normalization parameters
norm_obs = []
state = env.reset()[0]
for stp in range(pre_steps):
    action = env.action_space.sample()
    state, *others = env.step(action)
    norm_obs.append(state)
    
norm_obs_ = np.stack(norm_obs)
print('updating normalization parameters...')
agent.obs_rms.update(torch.tensor(norm_obs_).to('cuda'))

for ep in range(epoch, 10001):
# for ep in range(1):
    except_flag = False
    done = False
    try:
        state = env.reset()[0]
    except:
        ep -= 1
        continue
    ep_ext_reward = 0.0
    ep_int_reward = 0.0
    steps = 0
    # for i in range(20):
    while not done:
        action = agent.act(np.expand_dims(state, 0))
        try:
            next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
            env.render()
        except:
            except_flag = True
            break
        in_reward = agent.calc_intrin_rew(np.expand_dims(next_state, 0))
        agent.buffer.add_experience(state, action, next_state, reward, done, in_reward)
        state = next_state
        ep_ext_reward += reward
        ep_int_reward += in_reward
        steps += 1
        if steps >= max_steps:
            break
    if except_flag:
        ep -= 1
        continue
    print(f'ep. {ep}\tepisode_ext rewards: {ep_ext_reward}\tep_int_rewards: {ep_int_reward}')
    agent.write_plot_data(ep_ext_reward, ep_int_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_ext_reward)