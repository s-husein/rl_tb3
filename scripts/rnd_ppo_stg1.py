from gymenv import Gym
from algos import PPO
import numpy as np
from nets import make_dnn
import torch
import cv2 as cv

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

positions = [(3, 3)]
# angles = np.arange(0, 360, 45)
angles = [225, 270, 180, 135]
k_epochs = 10
batch_size = 512
pi_hid_layers = [512, 256]
min_batch_size = 4096
pi_conv_layers = [[16, 5, 1],
                  [32, 3, 1],
                  [64, 3, 1]]
episodes = 10000
lam = 0.95
gamma = 0.99
actor_lr = 3e-5
critic_lr = 7e-5
pred_lr= 1e-5
act_space = 'cont'
name = 'rnd_ppo_stg1_pc'
std_min_clip =  0.1
eps_clip= 0.2
beta = 0.07
max_pool = [2, 2]
max_steps = 15000
obs_scale_factor = 0.1

env = Gym(action_space=act_space, positions=positions, angles=angles, obs_scale_factor=obs_scale_factor, conv_layers=True)

pi = make_dnn(env, pi_hid_layers, act_space, 'actor', max_pool=max_pool, conv_layers=pi_conv_layers)
v = make_dnn(env, pi_hid_layers, act_space, net_type='critic', conv_layers = pi_conv_layers, max_pool=max_pool)

agent = PPO(k_epochs=k_epochs, batch_size=batch_size, min_batch_size=min_batch_size,conv_layer=True,
            actor_lr=actor_lr, critic_lr=critic_lr, act_space=act_space, name=name,
            lam=lam, std_min_clip=std_min_clip, beta=beta, eps_clip=eps_clip, gamma=gamma, actor=pi, critic=v)

epoch = agent.configs['epochs']

for ep in range(epoch, episodes+1):
# for ep in range(1):
    except_flag = False
    done = False
    try:
        state = env.reset()[0]
    except:
        ep -= 1
        continue
    ep_ext_reward = 0.0
    steps = 0
    # for i in range(15):
    while not done:
        d_s = (((np.transpose(state[0], (2, 0, 1))/255.0)-0.5)/0.5).astype(np.float32)
        action = agent.act(d_s)
        try:
            next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
            cv.imshow('depth', state[0])
            cv.waitKey(1)
        except:
            except_flag = True
            break
        d_ns = (((np.transpose(next_state[0], (2, 0, 1))/255.0)-0.5)/0.5).astype(np.float32)
        agent.buffer.add_experience(d_s, action, d_ns, reward, done)
        state = next_state
        ep_ext_reward += reward
        steps += 1
        if steps >= max_steps:
            break
    if except_flag:
        ep -= 1
        continue

    
    print(f'ep. {ep}\t{ep_ext_reward = :.3f}\t{steps = }')
    agent.write_plot_data(ep_ext_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_ext_reward)

