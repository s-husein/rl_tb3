from nets import make_dnn
from utils import Utils
from mems import Rollout
from gym import Env
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from paths import MODELFOLDER, PLOTFOLDER, REWARDFOLDER, STATUSFILE
from copy import deepcopy
from dists import MultiCategorical
from utils import RunningMeanStd
import os


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
                 net_is_shared = True, actor_lr = 0.00003, critic_lr = 0.0003, act_space = 'disc',
                 n_step_return = None, lam = None, gamma=0.99, std_min_clip = 0.08, conv_layers = None, max_pool = None,
                 beta = 0.03, bins = None, ordinal=False, act_fun = 'relu', policy_net = 'actor', value_net = 'critic'):
        
        self.buffer = Rollout()
        self.min_batch_size = min_batch_size
        self.net_is_shared = net_is_shared
        self.model_file = f'{MODELFOLDER}/{name}_model.pth'
        self.plot_file = f'{PLOTFOLDER}/{name}_plot.txt'
        self.reward_file = f'{REWARDFOLDER}/{name}_max_reward.txt'
        self.act_space = act_space
        self.max_rewards = self.check_rewards_file()
        self.n_step_ret = n_step_return
        self.lam = lam
        self.gamma = gamma
        self.std_min_clip = std_min_clip
        self.beta = beta
        if self.net_is_shared:
            self.model = make_dnn(env, hid_layers=hid_layer, net_type='shared', action_space=act_space,
                                  bins=bins, ordinal=ordinal, act_fn=act_fun, conv_layers=conv_layers, max_pool=max_pool).to(device)
            self.optim = Adam(self.model.parameters(), lr = actor_lr, eps = 1e-8)
            self.model.train()
            print(self.model)
        else:
            self.actor = make_dnn(env, net_type=policy_net, hid_layers=hid_layer, action_space=act_space,
                                  bins=bins, ordinal=ordinal, act_fn=act_fun, conv_layers=conv_layers, max_pool=max_pool).to(device)
            self.critic = make_dnn(env, net_type=value_net, hid_layers=hid_layer, action_space=act_space,
                                   bins=bins, ordinal=ordinal, act_fn=act_fun, conv_layers=conv_layers, max_pool=max_pool).to(device)
            self.act_optim = Adam(self.actor.parameters(), lr = actor_lr, eps = 1e-5)
            self.crit_optim = Adam(self.critic.parameters(), lr = critic_lr, eps = 1e-5)
            self.actor.train()
            self.critic.train()
            print(f'actor: {self.actor}\ncritic: {self.critic}')

    def act(self, state):
        state = torch.from_numpy(state).to(device)
        if self.net_is_shared:
            logits = self.model(state)[:-1]
        else:
            logits = self.actor(state)
        with torch.no_grad():
            if self.act_space == 'disc':
                dist = Categorical(logits=logits)
            elif self.act_space == 'cont':
                mean, std = torch.chunk(logits, 2)
                mean, std = F.tanh(mean), F.sigmoid(std).clip(self.std_min_clip, 0.7)
                dist = Normal(mean, std)
            action = dist.sample()
        return action.to(device)
    
    def calc_values(self, states):
        if self.net_is_shared:
            values = self.model(states)[:, -1]
        else:
            values = self.critic(states).transpose(0, 1)
        return values
    
    def calc_pd(self, states):
        if self.net_is_shared:
            logits = self.model(states)[:, :-1]
        else:
            logits = self.actor(states)
        return logits
    
    def adv_gae(self, values, next_values):
        rewards = self.buffer.traj['rewards']
        not_dones = 1 - np.array(self.buffer.traj['dones'])
        T = len(rewards)
        with torch.no_grad():
            advantage = torch.zeros_like(values, dtype=torch.float32).to(device)
        futureadv = 0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma*next_values[t]*not_dones[t] - values[t]
            futureadv = delta + self.gamma*self.lam*futureadv*not_dones[t]
            advantage[t] = futureadv
        target_values = (advantage + values).to(device)
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-08)
        return advantage, target_values
    
    def adv_nstep(self, values, next_values, n):
        rewards = self.buffer.traj['rewards']
        with torch.no_grad():
            rets = torch.zeros_like(values, dtype=torch.float32)
        future_ret = next_values[n-1]
        not_dones = 1 - np.array(self.buffer.traj['dones'])
        for t in reversed(range(n+1)):
            rets[t] = future_ret = rewards[t] + self.gamma*future_ret*not_dones[t]
        advs = (rets - values).to(device)
        target_values = rets.to(device)
        return advs, target_values
    
    def calc_adv(self, values, next_values):
        values = values.detach()
        if self.n_step_ret is not None:
            advs, target_values = self.adv_nstep(values, next_values, self.n_step_ret)
        elif self.lam is not None:
            advs, target_values = self.adv_gae(values, next_values)

        return advs, target_values
    
    def log_probs(self, logits, actions):
        if self.act_space == 'disc':
            dist = Categorical(probs = F.softmax(logits, dim=1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        elif self.act_space == 'cont':
            mean, std = torch.chunk(logits, 2, dim=1)
            mean, std = F.tanh(mean), F.sigmoid(std).clip(self.std_min_clip, 0.4)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)

        elif self.act_space == 'discretize':
            dist = MultiCategorical(probs = logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_probs, entropy
    
    def shared_loss(self, states, actions, next_states):
        logits = self.calc_pd(states)
        values = self.calc_values(states)
        with torch.no_grad():
            next_values = self.calc_values(next_states)
        log_probs, entropy = self.log_probs(logits, actions)
        advantages, target_values = self.calc_adv(values, next_values)
        self.optim.zero_grad()
        value_loss = F.mse_loss(values, target_values).to(device)
        policy_loss = ((-log_probs * advantages) - self.beta*entropy).mean().to(device)
        loss = value_loss+policy_loss
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.4)
        self.optim.step()
    
    def separate_loss(self, states, actions, next_states):
        logits = self.calc_pd(states)
        values = self.calc_values(states)
        with torch.no_grad():
            next_values = self.calc_values(next_states)
        log_probs, entropy = self.log_probs(logits, actions)
        advantages, target_values = self.calc_adv(values, next_values)
        
        policy_loss = ((-log_probs*advantages)-self.beta*entropy).mean().to(device)
        self.act_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.3)
        self.act_optim.step()

        value_loss = F.mse_loss(values, target_values).to(device)
        self.crit_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.crit_optim.step()

    def train(self):
        if self.buffer.size > self.min_batch_size:
            states = torch.stack(self.buffer.traj['states']).to(device)
            actions = torch.stack(self.buffer.traj['actions']).to(device)
            next_states = torch.stack(self.buffer.traj['next_states']).to(device)
            if self.net_is_shared:
                self.shared_loss(states, actions, next_states)
            else:
                self.separate_loss(states, actions, next_states)
            self.buffer.reset()
        else:
            pass


class PPO(A2C):
    def __init__(self, env: Env, k_epochs, batch_size = 256, hid_layer = [256, 256], conv_layers = None, max_pool = None, bins=None,
                 min_batch_size=2048, net_is_shared = False, actor_lr=0.0003, critic_lr = 0.001,
                 act_space = 'disc', name='ppo', lam=0.95, std_min_clip = 0.07,
                 beta=0.01, eps_clip=0.1, gamma=0.99, act_fn = 'relu', policy_net = 'actor', value_net = 'critic'):
        
        super(PPO, self).__init__(env= env, name = name, min_batch_size=min_batch_size, net_is_shared=net_is_shared,
                                  actor_lr=actor_lr, critic_lr=critic_lr, act_space=act_space, bins=bins, hid_layer=hid_layer,
                                  lam=lam, std_min_clip=std_min_clip, beta = beta, gamma=gamma, act_fun=act_fn,
                                  conv_layers = conv_layers, max_pool=max_pool,policy_net=policy_net, value_net=value_net)
        

        self.batch_size = batch_size
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

        if  self.net_is_shared:
            self.old_policy = deepcopy(self.model)
            assert id(self.old_policy) != id(self.model)
        else:
            self.old_policy = deepcopy(self.actor)
            assert id(self.old_policy) != id(self.actor)

    def act(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            logits = self.old_policy(state).squeeze()
            if self.net_is_shared:
                logits = logits[:-1]

            if self.act_space == 'disc':
                dist = Categorical(logits=logits)
            elif self.act_space == 'cont':
                mean, std = torch.chunk(logits, 2)
                mean, std = F.tanh(mean), F.sigmoid(std).clip(self.std_min_clip, 0.7)
                dist = Normal(mean, std)
            elif self.act_space == 'discretize':
                dist = MultiCategorical(probs=logits)
            
            action = dist.sample()
        return action.to(device)
    
    def shared_loss(self, states, actions, values, advs, tar_values):
        logits = self.calc_pd(states)
        log_probs, entropy = self.log_probs(logits, actions)
        with torch.no_grad():
            old_logits = self.old_policy(states)
            old_log_probs = self.log_probs(old_logits, actions)[0]
        assert old_log_probs.shape == log_probs.shape
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advs
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advs
        clip_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, tar_values).to(device)
        policy_loss = (clip_loss - self.beta*entropy.mean()).to(device)
        self.optim.zero_grad()
        loss = value_loss+policy_loss
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.4)
        self.optim.step()

    
    def separate_loss(self, states, actions, advs, tar_values, intr_tar_values=None):
        logits = self.calc_pd(states)
        log_probs, entropy = self.log_probs(logits, actions)
        with torch.no_grad():
            old_logits = self.old_policy(states)
            old_log_probs = self.log_probs(old_logits, actions)[0]
        assert old_log_probs.shape == log_probs.shape

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advs
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advs
        clip_loss = -torch.min(surr1, surr2).mean()
        policy_loss = (clip_loss - (self.beta*entropy).mean()).to(device)
        self.act_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.4)
        self.act_optim.step()

        intr_loss = 0
        values = self.calc_values(states)
        if intr_tar_values is not None:
            values, intr_values = values[0], values[1]
            intr_loss = F.mse_loss(intr_values, intr_tar_values).to(device)
        
        value_loss = F.mse_loss(values, tar_values).to(device)
        critic_loss = value_loss + intr_loss
        self.crit_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.4)
        self.crit_optim.step()    

    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        if self.net_is_shared:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.model.train()
            self.old_policy.load_state_dict(self.model.state_dict())
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.act_optim.load_state_dict(checkpoint['act_optim_state_dict'])
            self.crit_optim.load_state_dict(checkpoint['crit_optim_state_dict'])
            self.actor.train()
            self.critic.train()
            self.old_policy.load_state_dict(self.actor.state_dict())
        print('checkpoint loaded...')
        return checkpoint['epoch']

    def train(self):
        if self.buffer.size > self.min_batch_size:
            print('training...')
            states = torch.stack(self.buffer.traj['states']).to(device)
            actions = torch.stack(self.buffer.traj['actions']).to(device)
            next_states = torch.stack(self.buffer.traj['next_states']).to(device)
            with torch.no_grad():
                values = self.calc_values(states)
                next_values = self.calc_values(next_states)
            advs, tar_values = self.calc_adv(values, next_values)
            for _ in range(self.k_epochs):
                mini_batches = self.buffer.get_mini_batches(self.batch_size)
                for mini_batch in mini_batches:                    
                    min_states = min_actions = min_advs = min_tar_values = torch.zeros(len(mini_batch)).to(device)
                    min_states = torch.stack([states[ind] for ind in mini_batch]).to(device)
                    min_actions = torch.stack([actions[ind] for ind in mini_batch]).to(device)
                    min_advs = torch.tensor([advs[ind] for ind in mini_batch]).to(device)
                    min_tar_values = torch.tensor([tar_values[ind] for ind in mini_batch]).to(device)
                    if self.net_is_shared:
                        self.shared_loss(states=min_states, actions=min_actions, advs=min_advs, tar_values=min_tar_values)
                    else:
                        self.separate_loss(min_states, min_actions, min_advs, min_tar_values)                    
            self.buffer.reset()
            print('trained...')
            if self.net_is_shared:
                self.old_policy.load_state_dict(self.model.state_dict())
            else:
                self.old_policy.load_state_dict(self.actor.state_dict())
        else:
            pass



class RND_PPO(PPO):
    def __init__(self, env: Env, k_epochs, batch_size = 256, hid_layer = [256, 256], conv_layers = None, max_pool = None, bins=None,
                 min_batch_size=2048, net_is_shared = False, actor_lr=0.0003, critic_lr = 0.001, pred_lr = 0.001,
                 act_space = 'disc', name='ppo', lam=0.95, std_min_clip = 0.07, predictor_update=0.5, rnd_feat=[128],
                 beta=0.01, eps_clip=0.1, gamma_e=0.999, gamma_i = 0.99, act_fn = 'relu', ext_coef=2, intr_coef=1):
        
        super(RND_PPO, self).__init__(env=env, k_epochs=k_epochs, batch_size=batch_size, hid_layer=hid_layer, conv_layers=conv_layers,
                        max_pool=max_pool, bins=bins, min_batch_size=min_batch_size, net_is_shared=net_is_shared,
                        actor_lr=actor_lr, critic_lr=critic_lr, act_space=act_space, name=name, lam=lam,
                        std_min_clip=std_min_clip, beta=beta, eps_clip=eps_clip, gamma=gamma_e,
                        act_fn=act_fn, value_net='two_head', )
        
        self.gamma_i = gamma_i
        self.ext_coef=ext_coef
        self.intr_coef = intr_coef
        self.pred_update = predictor_update
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd()
        self.targ_net_file = f'{MODELFOLDER}/target_net.pth'
        rnd_hid_layer = hid_layer + rnd_feat
        self.targ_net = make_dnn(env, rnd_hid_layer, net_type='rnd',action_space=act_space,
                                 act_fn=act_fn, conv_layers=conv_layers, max_pool=max_pool).to(device)
        for param in self.targ_net.parameters():
            param.requires_grad = False

        self.check_targ_net_file()
        self.pred_net = make_dnn(env, rnd_hid_layer, net_type='rnd', action_space=act_space,
                                 act_fn=act_fn, conv_layers=conv_layers, max_pool=max_pool).to(device)
        self.pred_net.train()
        self.pred_net_optim = Adam(self.pred_net.parameters(), lr = pred_lr)
        print(f'predictor network: {self.pred_net}')
    
    def check_targ_net_file(self):
        if os.path.exists(self.targ_net_file):
            self.targ_net.load_state_dict(torch.load(self.targ_net_file))
        else:
            self.create_file(self.targ_net_file)
            torch.save(self.targ_net.state_dict(), self.targ_net_file)

    def calc_intrin_rew(self, next_state):
        next_state = torch.from_numpy(next_state).to(device)
        self.obs_rms.update(next_state)
        norm_obs = ((next_state - self.obs_rms.mean)/self.obs_rms.std).clip(-5, 5)
        with torch.no_grad():
            pred_feat = self.pred_net(norm_obs)
            targ_feat = self.targ_net(norm_obs)

        intrin_rew = (targ_feat - pred_feat).pow(2).sum(-1).detach()
        return intrin_rew.item()

    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        if self.net_is_shared:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.model.train()
            self.old_policy.load_state_dict(self.model.state_dict())
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.act_optim.load_state_dict(checkpoint['act_optim_state_dict'])
            self.crit_optim.load_state_dict(checkpoint['crit_optim_state_dict'])
            self.actor.train()
            self.critic.train()
            self.old_policy.load_state_dict(self.actor.state_dict())

        self.pred_net.load_state_dict(checkpoint['pred_net_state_dict'])
        self.pred_net_optim.load_state_dict(checkpoint['pred_net_optim'])
        self.pred_net.train()
        print('checkpoint loaded...')
        return checkpoint['epoch']

    def save_checkpoint(self, epoch, checkpath):
        file = open(STATUSFILE, 'w')
        if self.net_is_shared:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'pred_net_state_dict': self.pred_net.state_dict(),
                'pred_net_optim': self.pred_net_optim.state_dict(),
                'epoch': epoch
            }
        else:
             checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'act_optim_state_dict': self.act_optim.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'crit_optim_state_dict': self.crit_optim.state_dict(),
                'pred_net_state_dict': self.pred_net.state_dict(),
                'pred_net_optim': self.pred_net_optim.state_dict(),
                'epoch': epoch
            }
        file.write(checkpath)
        file.close()
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')

    def write_plot_data(self, ext_rewards, intr_rewards):
        self.write_file(self.plot_file, f'{ext_rewards},{intr_rewards}\n')

    def check_status_file(self):
        checkpath = self.read_file(STATUSFILE)
        epoch = 0
        if checkpath != '':
            epoch = self.load_checkpoint(checkpath) + 1
            file = open(self.plot_file, 'r')
            lines = file.readlines()
            file = open(self.plot_file, 'w')
            file.writelines(lines[:epoch+1])
            file.close()
        else:
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file, 'Extrinsic_Rewards,Intrinsic_Rewards\n')
            epoch = 0
        return epoch

    def calc_intr_adv(self, intrin_values, intrin_nxt_values):
        rewards_ = self.buffer.traj['in_rewards']
        T = len(rewards_)
        with torch.no_grad():
            rewards = torch.tensor(rewards_, dtype=torch.float32).to(device)
            advantage = torch.zeros_like(intrin_values, dtype=torch.float32).to(device)

        self.reward_rms.update(rewards)
        rewards = rewards/self.reward_rms.std
        futureadv = 0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma_i*intrin_nxt_values[t] - intrin_values[t]
            futureadv = delta + self.gamma_i*self.lam*futureadv
            advantage[t] = futureadv
        target_values = (advantage + intrin_values).to(device)
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-08)
        return advantage, target_values
    
    def train(self):
        if self.buffer.size > self.min_batch_size:
            print('training...')
            states = torch.stack(self.buffer.traj['states']).to(device)
            actions = torch.stack(self.buffer.traj['actions']).to(device)
            next_states = torch.stack(self.buffer.traj['next_states']).to(device)

            with torch.no_grad():
                values = self.calc_values(states)
                next_values = self.calc_values(next_states) 
            ext_val, intr_val = values[0], values[1]
            ext_nxt_val, intr_nxt_val = next_values[0], next_values[1]

            intr_advs, intr_tar_values = self.calc_intr_adv(intr_val, intr_nxt_val)
            ext_advs, ex_tar_values = self.calc_adv(ext_val, ext_nxt_val)

            advs = self.ext_coef*ext_advs + self.intr_coef*intr_advs

            for _ in range(self.k_epochs):
                mini_batches = self.buffer.get_mini_batches(self.batch_size)
                for mini_batch in mini_batches:                    
                    min_states = min_actions = min_advs = min_ex_tar_values = min_intr_tar_values, min_next_states = torch.zeros(len(mini_batch)).to(device)

                    min_states = torch.stack([states[ind] for ind in mini_batch]).to(device)
                    min_next_states = torch.stack([next_states[ind] for ind in mini_batch]).to(device)
                    min_actions = torch.stack([actions[ind] for ind in mini_batch]).to(device)
                    min_advs = torch.tensor([advs[ind] for ind in mini_batch]).to(device)
                    min_ex_tar_values = torch.tensor([ex_tar_values[ind] for ind in mini_batch]).to(device)
                    min_intr_tar_values = torch.tensor([intr_tar_values[ind] for ind in mini_batch]).to(device)
                    if self.net_is_shared:
                        #add the intrinsic target values to the shared loss function
                        self.shared_loss(states=min_states, actions=min_actions, advs=min_advs, tar_values=min_ex_tar_values)
                    else:
                        self.separate_loss(min_states, min_actions, min_advs, min_ex_tar_values, min_intr_tar_values)

                    normz_states = ((min_next_states-self.obs_rms.mean)/self.obs_rms.std).clip(-5, 5)
                    with torch.no_grad():
                        targ_feat = self.targ_net(normz_states).detach()
                    pred_feat = self.pred_net(normz_states)

                    frwd_loss = F.mse_loss(pred_feat, targ_feat, reduction='none').mean(dim=-1)
                    mask = torch.rand(len(frwd_loss)).to(device)
                    mask = (mask < self.pred_update).type(torch.FloatTensor).to(device)
                    frwd_loss = (frwd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
                    self.pred_net_optim.zero_grad()
                    frwd_loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.pred_net.parameters(), 0.4)
                    self.pred_net_optim.step()

            self.buffer.reset()
            print('trained...')
            if self.net_is_shared:
                self.old_policy.load_state_dict(self.model.state_dict())
            else:
                self.old_policy.load_state_dict(self.actor.state_dict())
        else:
            pass
        