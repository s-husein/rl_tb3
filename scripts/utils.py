from paths import CHECKPOINT_DIR, MISC_DIR
import torch
import os
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Utils:
    def __init__(self):
        self.model = None
        self.model_file = ''
        self.plot_file = ''
        self.net_type = ''
        self.max_rewards = 0

    def read_file(self, path):
        file = open(path, 'r')
        file.seek(0)
        info = file.readline()
        file.close()
        return info

    def write_file(self, path, content):
        mode = 'w'
        if path == self.plot_file:
            mode = '+a'
        file = open(path, mode=mode)
        file.write(content)
        file.close()

    def create_file(self, path):
        file = open(path, 'w')
        file.close()

    def create_checkpoint_file(self, num):
        path = f'{CHECKPOINT_DIR}/checkpoint_{num}.pth'
        file = open(path, 'w')
        file.close()
        return path
    
    def save_config(self, args: dict):
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(args, file)

    def check_status(self, episodes):
        status = self.configs['status']
        if status == 'finished':
            raise Exception('Training already finished...')
        elif status == 'not_started':
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file, 'Rewards\n')
            epoch = self.configs['epochs'] = 0
            self.configs['status'] = 'in_progress'
        elif status == 'in_progress':
            epoch = self.configs['epochs']
            if epoch+1 >= episodes:
                self.configs['status'] = 'finished'
                with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
                    yaml.safe_dump(self.configs, conf_file)
                return epoch+1
            checkpath = f'{CHECKPOINT_DIR}/checkpoint_{epoch}.pth'
            if os.path.exists(checkpath):
                self.load_checkpoint(checkpath)
                file = open(self.plot_file, 'r')
                lines = file.readlines()
                file = open(self.plot_file, 'w')
                file.writelines(lines[:epoch+1])
                file.close()
        return epoch+1
        

    def write_plot_data(self, rewards):
        self.write_file(self.plot_file, f'{rewards}\n')

    def save_plot(self):
        data = pd.read_csv(self.plot_file)
        with plt.style.context('default'):
            plt.figure(figsize=(11, 9))
            data.rolling(int(self.configs['epochs']/20)).mean().plot(color='red', linewidth=3)
            plt.legend().set_visible(False)
            plt.xlabel('Episodes', fontsize=13)
            plt.ylabel('Average Rewards', fontsize=13)
            plt.xticks(fontsize=13)
            plt.grid(linestyle='--')
            plt.savefig(f'{MISC_DIR}/plot.png', dpi=300)
            plt.close('all')
        del data

    def save_checkpoint(self, epoch, checkpath):
        if self.net_is_shared:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'epoch': epoch
            }
        else:
             checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'act_optim_state_dict': self.act_optim.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'crit_optim_state_dict': self.crit_optim.state_dict(),
                'epoch': epoch
            }
        self.configs['epochs'] = epoch
        with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
            yaml.safe_dump(self.configs, conf_file)
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')
    
    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        if self.net_is_shared:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.model.train()
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.act_optim.load_state_dict(checkpoint['act_optim_state_dict'])
            self.crit_optim.load_state_dict(checkpoint['crit_optim_state_dict'])
            self.actor.train()
            self.critic.train()
        print('checkpoint loaded...')
    
    def save_check_interval(self, episodes, epoch, interval=50, queue_size=20):
        if not(epoch % interval) and epoch > 0:
            if epoch >= episodes-1:
                self.configs['status'] = 'finished'
            checkpath = self.create_checkpoint_file(epoch)
            self.save_checkpoint(epoch, checkpath)
            self.save_plot()
            del_checkpoint = f'{CHECKPOINT_DIR}/checkpoint_{epoch - (queue_size*interval)}.pth'
            if os.path.exists(del_checkpoint):
                os.remove(del_checkpoint)

    
    def load_model(self):
        print('loading model...')
        if self.net_is_shared:
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
        else:
            model = torch.load(self.model_file)
            self.actor.load_state_dict(model['actor_state_dict'])
            self.critic.load_state_dict(model['critic_state_dict'])
            self.actor.eval()
            self.critic.eval()
        print('model loaded...')

    def save_model(self):
        if not self.net_is_shared:
            model = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict()
            }
            torch.save(model, self.model_file)
        else:
            torch.save(self.model.state_dict(), self.model_file)
        print('model saved...')

    def save_best_model(self, rewards):
        if rewards > self.configs['max_reward']:
            self.configs['max_reward'] = rewards
            self.save_model()
            with open(f'{MISC_DIR}/misc.yaml', 'w') as conf_file:
                yaml.safe_dump(self.configs, conf_file)

    def check_rewards_file(self):
        if os.path.exists(self.reward_file):
            reward = float(self.read_file(self.reward_file))
        else:
            self.create_file(self.reward_file)
            reward = -1000.0
            self.write_file(self.reward_file, f'{reward}')
        return reward


class RunningMeanStd:
    def __init__(self, epsilon=1e-3, shape = ()):
        self.eps = epsilon
        self.shape = shape
        self.reset()

    def reset(self):
        self.mean = torch.zeros(size= self.shape)
        self.var = torch.ones(size=self.shape)
        self.std = torch.sqrt(self.var) + self.eps
        self.count = self.eps

    def update(self, x: torch.Tensor):
        x = x.squeeze().detach()
        batch_mean = torch.mean(x).to(device)
        batch_var = torch.var(x).to(device)
        batch_count = x.shape[0]

        new_count = batch_count + self.count
        delta = batch_mean - self.mean
        new_mean = self.mean + (delta*batch_count)/new_count
        ma = self.var*self.count
        mb = batch_var*batch_count

        m2 = ma+mb + torch.square(delta) * self.count * batch_count / new_count

        new_var = m2/new_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        self.std = torch.sqrt(self.var) + self.eps
    
