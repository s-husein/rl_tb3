import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from paths import MODEL, STATUSFILE, PLOTFILE
import pickle as pik
import utils


pu = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {pu}')


class Actor(nn.Module):
    def _init_(self, env, hid_layers = [128, 128], is_disc = True):
        super().__init__()
        self.is_disc = is_disc
        self.in_feat = env.observation_space.shape[0]*env.observation_space.shape[1]
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softm = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        if self.is_disc:
            logits = self.action(x)
        else: logits = self.actor_u, self.actor_a
        return logits


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
    
class Agent():
    def __init__(self, lr= 0.00003):
        self.model = NeuralNet().to(device=pu)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

    def act(self, state):
        state = torch.from_numpy(state).to(pu)
        state = torch.flatten(state/255)
        value, lin_mean, ang_mean, stds = self.model(state)
        dist = MultivariateNormal(torch.tensor([lin_mean.item(), ang_mean.item()], device=pu), stds.diag())
        self.entropy.append(dist.entropy())
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        if self.name != 'reinforce':
            self.values.append(value)

        return (action[0].clip(0, 1).item()*0.22, action[1].clip(-1, 1).item()*0.5)
    
    def reset_params(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropy = []
    
    def check_status(self):
        data = utils.load_status_dict()
        epochs = 0
        if data['path'] != '':
            epochs = data['num']
            self.load_checkpoint(data['path'])
            print('checkpoint loaded...')
            file = open(PLOTFILE, 'r')
            lines = file.readlines()
            file = open(PLOTFILE, 'w')
            file.writelines(lines[:data['num']+1])
            file.close()
        else:
            file = open(PLOTFILE, 'w')
            file.close()

        if utils.read_file(PLOTFILE) == '':
            utils.write_file(PLOTFILE, 'Rewards,Steps\n')
        
        return epochs

    def save_checkpoint(self, epoch, checkpath):
        file = open(STATUSFILE, 'w')
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': epoch
        }
        file.write(checkpath)
        file.close()
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')

    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.model.train()
        return checkpoint['epoch']
    
    def save_model(self):
        torch.save(self.model.state_dict(), MODEL)
        print('model saved...')
    
    def load_model(self):
        self.model.load_state_dict(torch.load(MODEL))
        self.model.eval()
        print('loaded model..')

    def advantage_gae(self):
        futureadv = 0.0
        T = len(self.rewards)
        advantage = np.zeros_like(self.rewards)
        for t in reversed(range(T)):
            delta = self.rewards[t] + 0.999*self.values[t+1] - self.values[t]
            futureadv = delta + 0.999*0.7*futureadv
            advantage[t] = futureadv
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
        return advantage
    
    def advantage_nstep(self):
        futureret = 0.0
        T = len(self.rewards)
        ret = np.zeros_like(self.rewards)
        for t in reversed(range(T)):
            futureret = self.rewards[t] + 0.99*futureret
            ret[t] = futureret
        advantage = ret - self.values[:T]
        return (advantage - advantage.mean())/(advantage.std() + 1e-8)
    
    def a2c_train(self):
        advantage = torch.tensor(self.advantage_nstep(), device=pu)
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-07)
        log_prob = torch.stack(self.log_probs).to(pu)
        entropy = torch.stack(self.entropy).to(pu)
        value_loss = (advantage**2).mean()
        policy_loss = (-advantage*log_prob - 0.0*entropy).mean()
        loss = torch.add(value_loss, policy_loss)
        return loss
    
    def reinforce_train(self):
        g = 0.0
        T = len(self.rewards)
        ret = np.empty(T, dtype=np.float32)
        for t in reversed(range(T)):
            g = self.rewards[t] + g*0.99
            ret[t] = g
        ret = (ret - ret.mean())/(ret.std() + 1e-07)
        G = torch.tensor(ret, device=pu)
        log_probs = torch.stack(self.log_probs).to(device=pu)
        loss = (-log_probs * G).to(pu)
        loss = torch.sum(loss)
        return loss
    
    def train(self):
        loss = self.train_func[self.name]()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.2)
        self.optim.step()
        return loss.item()
    
class REINFORCE(Agent):
    def __init__(self, env, hid_layers, convution = False):
        super().__init__()
        
        self.model = NeuralNet(inp_lay = env.observation.shape[0]*env.observation.shape[1],shared=True)
        

    