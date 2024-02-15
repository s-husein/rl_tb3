import torch.nn as nn
import torch
from nets import make_dnn

class RND(nn.Module):
    def __init__(self, env, action_space, conv_layers = None, hid_layers = [64, 64], max_pool=None):
        super(RND, self).__init__()
        self.predictor = make_dnn(env, hid_layers=hid_layers, action_space=action_space, net_type='rnd',
                                  conv_layers=conv_layers, max_pool=max_pool)
        self.target = make_dnn(env, hid_layers=hid_layers, action_space=action_space, net_type='rnd',
                                  conv_layers=conv_layers, max_pool=max_pool)
        
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        pred_feat = self.predictor(x)
        with torch.no_grad():
            targ_feat = self.target(x)

        return pred_feat, targ_feat