import gym
from algos import REINFORCE



env = gym.make('LunarLander-v2', render_mode = 'rgb_array')

agent = REINFORCE(env=env, lr=0.0007)

for ep in range(5000):
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
        if steps >= 1000:
            break
    agent.train()
    print(f'ep. {ep}\tepisode rewards: {ep_reward}')