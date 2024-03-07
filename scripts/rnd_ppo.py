from gymenv import Gym
from algos import RND_PPO
import numpy as np
import torch
from algos import device

positions = [(1, -1), (1, -2)]

angles = np.arange(0, 360, 15)
max_steps = 10000
act_space = 'cont'
rnd_hid_layer = hid_layers = [256, 256, 128]

conv_layers = [[16, 5, 1],
               [32, 3, 1]]
max_pool = [2, 2]
pre_steps = 5 

env = Gym(action_space=act_space, positions=positions, angles=angles, obs_scale_factor=0.1, conv_layers=conv_layers)

agent = RND_PPO(env, k_epochs=4, batch_size=128, hid_layer=hid_layers, min_batch_size=2048, bins=7, conv_layers=conv_layers,
                actor_lr=0.00003, critic_lr=0.00007, pred_lr=0.0001, act_space=act_space, name='rnd_ppo',
                rnd_hid_layer=rnd_hid_layer, std_min_clip=0.1, eps_clip=0.1, beta=0.001,
                max_pool=max_pool, act_fn='elu', rnd_conv_layer=conv_layers)

epoch = agent.check_status_file()


print('taking random steps in environment for initialization of parameters...')
norm_obs = []
for stp in range(pre_steps):
    action = env.action_space.sample()
    state, *others = env.step(action)
    norm_obs.append(state)

    
norm_obs_ = np.stack(norm_obs)
print('updating normalization parameters...')
agent.obs_rms.update(torch.tensor(norm_obs_).to(device))

for ep in range(epoch, 50001):
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
        agent.buffer.add_experience(state, action, next_state, np.clip(reward, -1, 1), done, in_reward)
        state = next_state
        ep_ext_reward += reward
        ep_int_reward += in_reward
        steps += 1
        if steps >= max_steps:
            break
    if except_flag:
        ep -= 1
        continue
    print(f'ep. {ep}\t{ep_ext_reward = :.3f}\t{ep_int_reward = :.3f}\t{steps = }')
    agent.write_plot_data(ep_ext_reward, ep_int_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_ext_reward)