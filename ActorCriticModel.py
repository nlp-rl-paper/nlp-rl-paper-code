import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from vizdoom import *
from PPO_main_globals import DEVICE,ACTIONS

class ActorCriticModel(nn.Module):
    def __init__(self, output_len, word_embedding_dimension, sentence_max_size, hidden_units, textcnn_filter_count):
        super(ActorCriticModel, self).__init__()
        self.action_network = nn.Module()
        self.value_network = nn.Module()

        self.action_network.hidden_units = hidden_units
        self.action_network.out_channel = output_len
        self.action_network.label_num = output_len
        #con.action_networkvolution layers
        self.action_network.conv3 = nn.Conv2d(1, textcnn_filter_count, (3, word_embedding_dimension))
        self.action_network.conv4 = nn.Conv2d(1, textcnn_filter_count, (4, word_embedding_dimension))
        self.action_network.conv5 = nn.Conv2d(1, textcnn_filter_count, (5, word_embedding_dimension))
        self.action_network.conv8 = nn.Conv2d(1, textcnn_filter_count, (8, word_embedding_dimension))
        self.action_network.conv11 = nn.Conv2d(1, textcnn_filter_count, (11, word_embedding_dimension))
        #max.action_network pooling
        self.action_network.Max3_pool = nn.MaxPool2d((sentence_max_size - 3 + 1, 1))
        self.action_network.Max4_pool = nn.MaxPool2d((sentence_max_size - 4 + 1, 1))
        self.action_network.Max5_pool = nn.MaxPool2d((sentence_max_size - 5 + 1, 1))
        self.action_network.Max8_pool = nn.MaxPool2d((sentence_max_size - 8 + 1, 1))
        self.action_network.Max11_pool = nn.MaxPool2d((sentence_max_size - 11 + 1, 1))
        #fea.action_networkture extraction
        self.action_network.fc1 = nn.Linear(5 * textcnn_filter_count, hidden_units)
        self.action_network.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        #pol.action_networkicy estimator
        self.action_network.policy_layer = nn.Linear(hidden_units // 2,out_features=output_len)

        torch.nn.init.xavier_uniform_(self.action_network.conv3.weight)
        torch.nn.init.xavier_uniform_(self.action_network.conv4.weight)
        torch.nn.init.xavier_uniform_(self.action_network.conv5.weight)
        torch.nn.init.xavier_uniform_(self.action_network.conv8.weight)
        torch.nn.init.xavier_uniform_(self.action_network.conv11.weight)
        torch.nn.init.xavier_uniform_(self.action_network.fc1.weight)
        torch.nn.init.xavier_uniform_(self.action_network.fc2.weight)
        torch.nn.init.uniform_(self.action_network.policy_layer.weight)

        self.value_network.hidden_units = hidden_units
        self.value_network.out_channel = output_len
        self.value_network.label_num = output_len
        # convaluen_networkvolution layers
        self.value_network.conv3 = nn.Conv2d(1, textcnn_filter_count, (3, word_embedding_dimension))
        self.value_network.conv4 = nn.Conv2d(1, textcnn_filter_count, (4, word_embedding_dimension))
        self.value_network.conv5 = nn.Conv2d(1, textcnn_filter_count, (5, word_embedding_dimension))
        self.value_network.conv8 = nn.Conv2d(1, textcnn_filter_count, (8, word_embedding_dimension))
        self.value_network.conv11 = nn.Conv2d(1, textcnn_filter_count, (11, word_embedding_dimension))
        # maxvaluen_network pooling
        self.value_network.Max3_pool = nn.MaxPool2d((sentence_max_size - 3 + 1, 1))
        self.value_network.Max4_pool = nn.MaxPool2d((sentence_max_size - 4 + 1, 1))
        self.value_network.Max5_pool = nn.MaxPool2d((sentence_max_size - 5 + 1, 1))
        self.value_network.Max8_pool = nn.MaxPool2d((sentence_max_size - 8 + 1, 1))
        self.value_network.Max11_pool = nn.MaxPool2d((sentence_max_size - 11 + 1, 1))
        # feavaluen_networkture extraction
        self.value_network.fc1 = nn.Linear(5 * textcnn_filter_count, hidden_units)
        self.value_network.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        # valvaluen_networkue estimator
        self.value_network.value_layer = nn.Linear(hidden_units // 2, out_features=1)
        # polvaluen_networkicy estimator

        torch.nn.init.xavier_uniform_(self.value_network.conv3.weight)
        torch.nn.init.xavier_uniform_(self.value_network.conv4.weight)
        torch.nn.init.xavier_uniform_(self.value_network.conv5.weight)
        torch.nn.init.xavier_uniform_(self.value_network.conv8.weight)
        torch.nn.init.xavier_uniform_(self.value_network.conv11.weight)
        torch.nn.init.xavier_uniform_(self.value_network.fc1.weight)
        torch.nn.init.xavier_uniform_(self.value_network.fc2.weight)
        torch.nn.init.uniform_(self.value_network.value_layer.weight)

    def forward(self, x):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        x = x.type(torch.cuda.FloatTensor)
        x_input = x
        batch = x.shape[0]

        # Convolution
        x1 = F.relu(self.value_network.conv3(x))
        x2 = F.relu(self.value_network.conv4(x))
        x3 = F.relu(self.value_network.conv5(x))
        x4 = F.relu(self.value_network.conv8(x))
        x5 = F.relu(self.value_network.conv11(x))

        # Pooling
        x1 = self.value_network.Max3_pool(x1)
        x2 = self.value_network.Max4_pool(x2)
        x3 = self.value_network.Max5_pool(x3)
        x4 = self.value_network.Max8_pool(x4)
        x5 = self.value_network.Max11_pool(x5)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3, x4, x5), -1)
        x = x.view(batch, -1)
        x = F.relu(self.value_network.fc1(x)) #feature extraction
        x = F.relu(self.value_network.fc2(x))  # feature extraction

        value = self.value_network.value_layer(x)

        x = x_input

        # Convolution
        x1 = F.relu(self.action_network.conv3(x))
        x2 = F.relu(self.action_network.conv4(x))
        x3 = F.relu(self.action_network.conv5(x))
        x4 = F.relu(self.action_network.conv8(x))
        x5 = F.relu(self.action_network.conv11(x))

        # Pooling
        x1 = self.action_network.Max3_pool(x1)
        x2 = self.action_network.Max4_pool(x2)
        x3 = self.action_network.Max5_pool(x3)
        x4 = self.action_network.Max8_pool(x4)
        x5 = self.action_network.Max11_pool(x5)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3, x4, x5), -1)
        x = x.view(batch, -1)
        x = F.relu(self.action_network.fc1(x))  # feature extraction
        x = F.relu(self.action_network.fc2(x))  # feature extraction

        pi = Categorical(logits=self.action_network.policy_layer(x))
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
    state = torch.from_numpy(state).float().to(DEVICE)
    action_probs = model(state)
    dist = action_probs[0]
    action = dist.sample()

    memory.states.append(state)
    memory.actions.append(action)
    memory.logprobs.append(dist.log_prob(action))

    return action