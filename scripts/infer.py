from gym_node import Gym
import torch.nn as nn
import torch 
import cv2 as cv

pu = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(624, 512)
        self.l2 = nn.Linear(512, 256)
        # self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = self.softmax(self.l4(x))
        return x

pos = [(0, 0)]
ang = [90]
env = Gym(positions = pos, disc_action=True, angles=ang)

state = env.reset()[0]
agent = NeuralNet().to(pu)
agent.load_state_dict(torch.load('/home/user/fyp_results/REINFORCE_TM_HV.pth'))
agent.eval()




for i in range(15):
    state = env.reset()[0]
    done = False
    while not done:
        action = torch.argmax(agent(torch.flatten(torch.tensor(state/255, dtype=torch.float32).to(pu))))
        obs, re,*others = env.step(action)
        state = obs
        cv.imshow('state', state)
        cv.waitKey(1)
    
cv.destroyAllWindows()
    