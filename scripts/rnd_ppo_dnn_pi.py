from gymenv import Gym
from algos import RND_PPO
import numpy as np
import torch
from algos import device
from utils import conv_params

angles = np.arange(0, 360, 15)

params = conv_params(positions = [(-9.5, 3.5)],           
                    k_epochs = 10,
                    batch_size = 128,
                    rnd_hid_layer = [512, 128],
                    hid_layer = [512, 256, 128],
                    min_batch_size = 4096,
                    conv_layers = [[16, 3, 1],
                                   [32, 3, 1]],
                    actor_lr = 3e-6,
                    critic_lr = 7e-5,
                    pred_lr= 1e-5,
                    action_space = 'cont',
                    name = 'rnd_ppo_rgb_1',
                    std_min_clip =  0.08,
                    eps_clip= 0.4,
                    beta = 0.01,
                    max_pool = [2, 2],
                    max_steps = 18000,
                    pre_steps = 200,
                    act_fn = 'relu',
                    ext_coef = 2,
                    obs_scale_factor = 0.1)


env = Gym(action_space=params['action_space'], positions=params['positions'],
          angles=angles, obs_scale_factor=params['obs_scale_factor'], conv_layers=params['conv_layers'])

agent = RND_PPO(env, k_epochs=params['k_epochs'], batch_size=params['batch_size'], hid_layer=params['hid_layer'], min_batch_size=params['min_batch_size'],
                actor_lr=params['actor_lr'], critic_lr=params['critic_lr'], pred_lr=params['pred_lr'], act_space=params['action_space'], name=params['name'],
                rnd_hid_layer=params['rnd_hid_layer'], std_min_clip=params['std_min_clip'], eps_clip=params['eps_clip'], beta=params['beta'],
                max_pool=params['max_pool'], act_fn=params['act_fn'], rnd_conv_layer=params['conv_layers'], ext_coef=params['ext_coef'])


agent.save_config(params)

epoch = agent.check_status_file()


print('taking random steps in environment for initialization of parameters...')
norm_obs = []
for stp in range(params['pre_steps']):
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
    # for i in range(25):
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
        if steps >= params['max_steps']:
            break
    if except_flag:
        ep -= 1
        continue
    print(f'ep. {ep}\t{ep_ext_reward = :.3f}\t{ep_int_reward = :.3f}\t{steps = }')
    agent.write_plot_data(ep_ext_reward, ep_int_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_ext_reward)