from gymenv import Gym
from algos import A2C, PPO


positions = [(-0.5, 0.5), (8, 0.5), (8.5, -8.5), (-0.5, -8.5)]
angles = [0, -90, 45, -45, 225, -225, 90, 180]

env = Gym(disc_action=False, positions=positions, angles=angles)


agent = PPO(env=env, k_epochs=12, net_type='actor-critic',
            name='ppo_256_256', act_space='cont', min_batch_size=2048,
            batch_size=256, lr=0.00007, hid_layer=[256, 256], std_min_clip=0.08, eps_clip=0.2)

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
        if steps >= 10000:
            break
    if except_flag:
        ep -= 1
        continue
    print(f'ep. {ep}\tepisode rewards: {ep_reward}')
    agent.write_plot_data(ep_reward)
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(ep_reward)