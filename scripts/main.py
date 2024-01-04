from scripts.old_algos import Agent
import gym
import cv2 as cv
import utils
from paths import 


env = Gym()
agent = Agent('a2c')

epoch = 0

epoch = agent.check_status()
    

for ep in range(epoch, 20000):
# for ep in range(1):
    flag = False
    done = False
    try:
        state = env.reset()
    except:
        ep -= 1
        continue
    steps = 0
    # for i in range(3):
    while not done:
        action = agent.act(state)
        try:
            state, reward, done = env.step(action)
        except:
            flag = True
            break
        agent.rewards.append(reward)
        steps += 1
        cv.imshow('state', state)
        cv.waitKey(1)
    if flag == True:
        agent.reset_params()
        ep -= 1
        continue
    if agent.name != 'reinforce':
        agent.get_value(state)
    loss = agent.train()
    tot_re = sum(agent.rewards)
    agent.reset_params()
    print(f'{ep}. steps: {steps} reward: {tot_re} loss: {loss}')
    write_file(PLOTFILE, f'{steps},{tot_re}\n')
    
    if ep%25 == 0 and ep != 0:
        checkpath = create_file(ep)
        agent.save_checkpoint(ep, checkpath)

    if steps >= 80000:
        agent.save_model()
cv.destroyAllWindows()