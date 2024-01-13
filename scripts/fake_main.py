import gym
from algos import A2C



env = gym.make('Pendulum-v1', render_mode = 'rgb_array')

agent = A2C(env=env, min_batch_size=128, lr=0.0007, act_space='cont', net_type='actor-critic', hid_layer=[64,64], gae_adv=True)

epoch = agent.check_status_file()

for ep in range(epoch, 10000):
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