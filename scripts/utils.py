from paths import CHECKPOINTFOLDER, STATUSFILE
import torch

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

    def create_checkpoint_file(self, num):
        path = f'{CHECKPOINTFOLDER}/checkpoint_{num}.pth'
        file = open(path, 'w')
        file.close()
        return path

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
            return epoch
        else:
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file, 'Rewards\n')
            epoch = 0

    def write_plot_data(self, rewards):
        self.write_file(self.plot_file, f'{rewards}\n')

    def save_checkpoint(self, epoch, checkpath):
        file = open(STATUSFILE, 'w')
        if self.net_type == 'shared':
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
        file.write(checkpath)
        file.close()
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')
    
    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        if self.net_type == 'shared':
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
        return checkpoint['epoch']
    
    def save_check_interval(self, epoch, interval=50):
        if not(epoch % interval):
            checkpath = self.create_checkpoint_file(epoch)
            self.save_checkpoint(epoch, checkpath)
    
    def load_model(self):
        print('loading model...')
        if self.shared_net:
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
        if self.net_type != 'shared':
            model = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict()
            }
            torch.save(model, self.model_file)
        else:
            torch.save(self.model.state_dict(), self.model_file)
        print('model saved...')

    def save_best_model(self, rewards):
        if rewards > self.max_rewards:
            self.max_rewards = rewards
            self.save_model()