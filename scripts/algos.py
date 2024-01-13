from nets import make_dnn
from utils import Utils
from mems import Rollout
from gym import Env
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
from paths import MODELFOLDER, PLOTFOLDER
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')

class REINFORCE(Utils):
    def __init__(self, env: Env, name='reinforce', hid_layer = [128, 128], net_type='shared', lr = 0.00003, act_space = 'disc'):
        self.buffer = Rollout()
        self.net_type = net_type
        self.model_file = f'{MODELFOLDER}/{name}_model.pth'
        self.plot_file = f'{PLOTFOLDER}/{name}_plot.txt'
        self.act_space = act_space
        self.check_status_file()
        self.model = make_dnn(env, net_type=self.net_type, action_space=act_space)
        self.optim = Adam(self.model.parameters(), lr = lr, eps = 1e-8)
        self.model.train()

    def act(self, state):
        state = torch.from_numpy(state)
        probs = F.softmax(self.model(state))
        if self.act_space == 'disc':
            dist = Categorical(probs=probs)
        action = dist.sample()
        return action
    

    def discounted_rewards(self, rewards, dones):
        g = 0.0
        not_dones = 1 - np.array(dones)
        T = len(rewards)
        ret = np.empty(T, dtype=np.float32)
        for t in reversed(range(T)):
            g = rewards[t] + g*0.999*not_dones[t]
            ret[t] = g
        ret = (ret - ret.mean())/ret.std()
        return ret
    

    def train(self):
        self.write_plot_data(sum(self.buffer.traj['rewards']))
        batch =  self.buffer.sample()
        states = torch.stack([self.buffer.traj['states'][i] for i in batch])
        actions = torch.stack([self.buffer.traj['actions'][i] for i in batch])
        probs = F.softmax(self.model(states))
        adv = torch.tensor(self.discounted_rewards(rewards=[self.buffer.traj['rewards'][i] for i in batch], dones=[self.buffer.traj['dones'][i] for i in batch]))
        if self.act_space == 'disc':
            dist = Categorical(probs=probs)

        log_probs = dist.log_prob(actions)
        loss = (-log_probs * adv).sum()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()
        self.buffer.reset()


class A2C(Utils):
    def __init__(self, env: Env, name='a2c', min_batch_size=128, hid_layer = [128, 128],
                 net_type='shared', lr = 0.00003, act_space = 'disc', n_step_return = None, gae_adv = False):
        self.buffer = Rollout()
        self.min_batch_size = min_batch_size
        self.net_type = net_type
        self.model_file = f'{MODELFOLDER}/{name}_model.pth'
        self.plot_file = f'{PLOTFOLDER}/{name}_plot.txt'
        self.act_space = act_space
        self.max_rewards = 0
        self.gae_adv = gae_adv
        self.n_step_ret = n_step_return
        if self.net_type == 'shared':
            self.model = make_dnn(env, hid_layers=hid_layer, net_type=self.net_type, action_space=act_space).to(device)
            self.optim = Adam(self.model.parameters(), lr = lr, eps = 1e-8)
            self.model.train()
        elif self.net_type == 'actor-critic':
            self.actor = make_dnn(env, net_type='actor', hid_layers=hid_layer, action_space=act_space).to(device)
            self.critic = make_dnn(env, net_type='critic', hid_layers=hid_layer, action_space=act_space).to(device)
            self.act_optim = Adam(self.actor.parameters(), lr = lr, eps = 1e-8)
            self.crit_optim = Adam(self.critic.parameters(), lr = lr, eps = 1e-8)
            self.actor.train()
            self.critic.train()

    def act(self, state):
        state = torch.from_numpy(state).to(device)
        if self.net_type == 'shared':
            logits = self.model(state)[:-1]
        else:
            logits = self.actor(state)
        with torch.no_grad():
            if self.act_space == 'disc':
                dist = Categorical(logits=logits)
            elif self.act_space == 'cont':
                mean, std = torch.chunk(logits, 2)
                mean, std = F.tanh(mean), F.sigmoid(std).clip(0.01, 0.3)
                dist = MultivariateNormal(mean, std.diag_embed())
            action = dist.sample()
        return action.to(device)
    
    def adv_gae(self, values, next_values):
        rewards = self.buffer.traj['rewards']
        not_dones = 1 - np.array(self.buffer.traj['dones'])
        T = len(rewards)
        advantage = np.zeros_like(rewards, dtype=np.float32)
        futureadv = 0
        for t in reversed(range(T)):
            delta = rewards[t] + 0.99*next_values[t]*not_dones[t] - values[t]
            futureadv = delta + 0.99*0.8*futureadv*not_dones[t]
            advantage[t] = futureadv
        with torch.no_grad():
            torch.tensor(advantage)
            target_values = advantage + values
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-08)
        return advantage, target_values
    
    def adv_nstep(self, values, next_values, n):
        rewards = self.buffer.traj['rewards']
        rets = torch.zeros_like(values, dtype=torch.float32)
        future_ret = next_values[n-1]
        not_dones = 1 - np.array(self.buffer.traj['dones'])
        for t in reversed(range(n+1)):
            rets[t] = future_ret = rewards[t] + 0.99*future_ret*not_dones[t]
        with torch.no_grad():
            advs = (rets - values).to(device)
            target_values = rets.to(device)
        return advs, target_values
    
    def calc_adv(self, values, next_values):
        if self.n_step_ret is not None:
            advs, target_values = self.adv_nstep(values, next_values, self.n_step_ret)
        elif self.gae_adv:
            advs, target_values = self.adv_gae(values, next_values)

        return advs, target_values
    
    def log_probs(self, logits, actions):
        if self.act_space == 'disc':
            dist = Categorical(probs = F.softmax(logits))
            log_probs = dist.log_prob(actions)

        elif self.act_space == 'cont':
            mean, std = torch.chunk(logits, 2, dim=1)
            mean, std = F.tanh(mean), F.sigmoid(std).clip(0.08, 0.3)
            dist = MultivariateNormal(mean, std.diag_embed())
            log_probs = dist.log_prob(actions)

        return log_probs
    
    def shared_loss(self, states, actions, next_states):
        logits = self.model(states)
        with torch.no_grad():
            next_values = self.model(next_states)[:, -1]
        dist_logits, values = logits[:, :-1], logits[:, -1]
        log_probs = self.log_probs(dist_logits, actions)
        advantages, target_values = self.calc_adv(values, next_values)
        self.optim.zero_grad()
        value_loss = F.mse_loss(values, target_values).to(device)
        policy_loss = -(log_probs * advantages).mean().to(device)
        loss = value_loss+policy_loss
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()
    
    def separate_loss(self, states, actions, next_states):
        logits, values = self.actor(states), torch.cat(torch.unbind(self.critic(states)))
        with torch.no_grad():
            next_values =  torch.cat(torch.unbind(self.critic(next_states)))
        log_probs = self.log_probs(logits, actions)
        advantages, target_values = self.calc_adv(values, next_values)

        policy_loss = -(log_probs*advantages).mean().to(device)
        value_loss = F.mse_loss(values, target_values).to(device)

        self.act_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.4)
        self.act_optim.step()

        self.crit_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.4)
        self.crit_optim.step()


    def train(self):
        if self.buffer.size > self.min_batch_size:
            states = torch.stack(self.buffer.traj['states']).to(device)
            actions = torch.stack(self.buffer.traj['actions']).to(device)
            next_states = torch.stack(self.buffer.traj['next_states']).to(device)

            if self.net_type == 'shared':
                self.shared_loss(states, actions, next_states)
            elif self.net_type == 'actor-critic':
                self.separate_loss(states, actions, next_states)

        else:
            pass

    





