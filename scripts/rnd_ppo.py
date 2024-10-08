from gymenv import Gym
from algos import PPO
import numpy as np
from nets import make_dnn
import torch
import cv2 as cv

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

positions = [(2, 2.5), (-2.5, 2.5), (-2.5, -0.5),
             (-2, -2.5), (2.5, 0.5), (2.5, -2.5), (1.5, -1.5)]
angles = np.arange(0, 360, 15)
k_epochs = 8
batch_size = 128
pi_hid_layers = [128, 128]
min_batch_size =2048
pi_conv_layers = [[16, 3, 2],
                  [32, 1, 1]]
rnd_conv_layers = [[16, 3, 1],
                   [32, 3, 1]]
rnd_hid_layers = [128, 64, 16]
actor_lr = 3e-6
critic_lr = 7e-5
pred_lr= 1e-5
act_space = 'cont'
name = 'rnd_ppo_5'
std_min_clip =  0.1
eps_clip= 0.2
beta = 0.001
max_pool = [2, 2]
max_steps = 10000
pre_steps = 200
act_fn = 'elu'
ext_coef = 2
obs_scale_factor = 0.1

env = Gym(action_space=act_space, positions=positions, angles=angles, obs_scale_factor=obs_scale_factor, conv_layers=True)

pi = make_dnn(env, pi_hid_layers, act_space, 'actor', max_pool=max_pool, conv_layers=pi_conv_layers)
v = make_dnn(env, pi_hid_layers, act_space, net_type='two_head', conv_layers = pi_conv_layers, max_pool=max_pool)
pred = make_dnn(env, rnd_hid_layers, net_type='rnd', conv_layers=rnd_conv_layers, max_pool=max_pool, img_type='rgb').to(pu)
#add the models to cuda
print(pred)

d, r = env.get_observation()
r = np.transpose(r, (2, 0, 1))
d_ = torch.tensor(r, dtype=torch.float32).to(pu).unsqueeze(0)

out = pred(d_)
print(out)






# agent = RND_PPO(env, k_epochs=k_epochs, batch_size=batch_size, hid_layer=hid_layers, min_batch_size=min_batch_size, conv_layers=conv_layers,
#                 actor_lr=actor_lr, critic_lr=critic_lr, pred_lr=pred_lr, act_space=act_space, name=name,
#                 rnd_hid_layer=rnd_hid_layer, std_min_clip=std_min_clip, eps_clip=eps_clip, beta=beta,
#                 max_pool=max_pool, act_fn=act_fn, rnd_conv_layer=conv_layers, ext_coef=ext_coef)

# agent.save_config(k_epochs=k_epochs, batch_size = batch_size, rnd_hid_layer=hid_layers, hid_layer = hid_layers,
#             min_batch_size = min_batch_size, conv_layers=conv_layers, actor_lr = actor_lr, critic_lr = critic_lr,
#             pred_lr = pred_lr, action_space = act_space, name=name, std_min_clip = std_min_clip, eps_clip = eps_clip, obs_scale_factor=obs_scale_factor,
#             beta=beta, max_pool = max_pool, max_steps=max_steps, pre_steps=pre_steps, activation_fun= act_fn, ext_coef = ext_coef)

# epoch = agent.check_status_file()


# for ep in range(epoch, 50001):
# # for ep in range(1):
#     except_flag = False
#     done = False
#     try:
#         state = env.reset()[0]
#     except:
#         ep -= 1
#         continue
#     ep_ext_reward = 0.0
#     ep_int_reward = 0.0
#     steps = 0
#     # for i in range(20):
#     while not done:
#         action = agent.act(np.expand_dims(state, 0))
#         try:
#             next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
#             env.render()
#         except:
#             except_flag = True
#             break
#         in_reward = agent.calc_intrin_rew(np.expand_dims(next_state, 0))
#         agent.buffer.add_experience(state, action, next_state, reward, done, in_reward)
#         state = next_state
#         ep_ext_reward += reward
#         ep_int_reward += in_reward
#         steps += 1
#         if steps >= max_steps:
#             break
#     if except_flag:
#         ep -= 1
#         continue
#     print(f'ep. {ep}\t{ep_ext_reward = :.3f}\t{ep_int_reward = :.3f}\t{steps = }')
#     agent.write_plot_data(ep_ext_reward, ep_int_reward)
#     agent.train()
#     agent.save_check_interval(epoch = ep)
#     agent.save_best_model(ep_ext_reward)