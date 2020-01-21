import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from vizdoom import *


class ActorCriticModel(nn.Module):
    def __init__(self, output_len, hidden_units, filter_count,resolution,n_channels,device):
        super(ActorCriticModel, self).__init__()
        self.device = device

        self.dim0 = ((((resolution[0] - 6) // 3 + 1) - 3) // 2 + 1)
        self.dim1 = ((((resolution[1] - 3) // 3 + 1) - 3) // 2 + 1)

        # -----ACTION NETWORK-----#
        self.hidden_units = hidden_units
        self.out_channel = output_len
        self.label_num = output_len
        #con.action_networkvolution layers
        self.conv1 = nn.Conv2d(n_channels, filter_count, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(filter_count, 2*filter_count, kernel_size=3, stride=2)

        #fea.action_networkture extraction
        self.fc1 = nn.Linear(self.dim1 * self.dim0 * 2*filter_count, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        #pol.action_networkicy estimator
        self.policy_layer = nn.Linear(hidden_units // 2,out_features=output_len)
        # valvaluen_networkue estimator
        self.value_layer = nn.Linear(hidden_units // 2, out_features=1)



    def forward(self, x):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        x = x.type(torch.cuda.FloatTensor)
        x_input = x

        x.to(self.device)
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_layer(x)
        pi = Categorical(logits=self.policy_layer(x))
        return pi,value

    @property
    def output_size(self):
        return self.hidden_units


    @property
    def is_recurrent(self):
        return False


    @property
    def recurrent_hidden_state_size(self):
        return 1

    def act(self, inputs, masks, deterministic=False):
        #value, actor_features,  = self.base(inputs, , masks)
        dist,value = self(inputs)
        #dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        _,value = self(inputs)
        return value

    def evaluate_actions(self, inputs , masks, action):
        dist,value  = self(inputs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


def evaluate(model : nn.Module , state, action):
    action_probs,value = model(state)
    dist = action_probs
    action_logprobs = dist.log_prob(action)
    dist_entropy = dist.entropy()
    state_value = value
    return action_logprobs, torch.squeeze(state_value), dist_entropy


def act(model : nn.Module ,state, memory):
    state = torch.from_numpy(state).float().to("cuda")
    action_probs = model(state)
    dist = action_probs[0]
    action = dist.sample()

    memory.states.append(state)
    memory.actions.append(action)
    memory.logprobs.append(dist.log_prob(action))

    return action