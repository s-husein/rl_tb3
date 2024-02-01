from gymenv import Gym
from algos import A2C, PPO
import numpy as np

positions = [(1, -1), (1, -2), (4, -1), (3, -1), (3, -2), (4, -2), (5, -1), (5, -3), (1, -4),
             (1, -5), (1, -6), (1, -8), (1, -10), (2, -10), (2, -8), (2, -6), (3, -9), (3, -8),
             (3, -7), (4, -8), (5, -7), (4, -10), (5, -10), (3, -4), (4, -4), (3, -5), (4, -5), (4, -6),
             (5, -7), (6, -7), (7, -9), (7, -10), (7, -7)]
angles = np.arange(0, 360, 15)
max_steps = 10000
act_space = 'cont'
env = Gym(action_space=act_space, positions=positions, angles=angles)


agent = PPO(env=env, k_epochs=10, net_is_shared=False,
            name='ppo_cont:256x256, batch_size: 64, lam: 0.95, gamma:0.99, net_type:sep',
            act_space=act_space, min_batch_size=2048, batch_size=64, actor_lr=0.0003, critic_lr=0.0007,
            lam=0.95, hid_layer=[256, 256], std_min_clip=0.08, eps_clip=0.3)

epoch = agent.check_status_file()

for ep in range(epoch, 50001):
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
        action = agent.act(state)
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