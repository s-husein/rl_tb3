import gym
from algos import A2C



env = gym.make('BipedalWalker-v3', render_mode = 'rgb_array')

agent = A2C(env=env, min_batch_size=256, lr=0.0007, act_space='cont', net_type='actor-critic')

epoch = agent.check_status_file()

for ep in range(epoch, 5000):
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
        if steps >= 1600:
            break
    agent.train()
    agent.save_check_interval(epoch = ep)
    agent.save_best_model(300)
    print(f'ep. {ep}\tepisode rewards: {ep_reward}')