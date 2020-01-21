#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello
# August 2017

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import os
import torch
import torch.nn as nn
from GloveEmbedder import GloveEmbedder
from AnacdoteParser import AnacdoteParser
from StoryParser import StoryParser
from ManyPatchStoryParser import StoryParser as PatchStoryParser
from DoomTextCNN import TextCNN
from PlotGenerator import PlotGenerator
from ArgumentParser import parse_arguments
from torch.autograd import Variable
import random as rnd
from tqdm import trange
import sys
from ReplayMemory import ReplayMemory

# Q-learning settings
#LEARNING_RATE = 0.00005
DISCOUNT_FACTOR = 0.9999
#EPOCHS = 50
#LEARNING_STEPS_PER_EPOCH = 100
REPLAY_MEMORY_SIZE = 10000
TEST_EPISODES_PER_EPOCH = 5
EPISODES_TO_WATCH = 15
WORD_EMBEDDING_LEN = 50


# EPS_DECAY_10 = 0.25
# EPS_DECAY_60 = 0.7

EPS_DECAY_10 = 0.1
EPS_DECAY_60 = 0.6

SAVE_MODEL = False
LOAD_MODEL = False
SKIP_LEARNING = False
FIGURE_BLOCK = False
SKIP_TEST=False


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype(np.float32)
    return img



def learn(s1, target_q):
    s1 = torch.from_numpy(s1).cuda(device)
    target_q = torch.from_numpy(target_q).cuda(device)
    s1, target_q = Variable(s1), Variable(target_q)
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


#need to reimplemeent using target network
def get_q_values(state):
    state = torch.from_numpy(state)#.type(torch.cuda.FloatTensor)
    state = state.to(device).type(torch.cuda.FloatTensor)
    state = Variable(state)
    return target_model(state)


def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.cpu().data.numpy()[0]
    return action,q.data

def get_double_network_q_values(state):
    #q_target[s',argmax_action_q[s',a']]
    state = torch.from_numpy(state)#.type(torch.cuda.FloatTensor)
    state.to(device)
    state = Variable(state)
    q = model(state).cpu().data.numpy()
    best_action_index = np.argmax(q,axis=1)
    q_target = target_model(state).cpu().data.numpy()
    return q_target[np.arange(len(best_action_index)),best_action_index]


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > BATCH_SIZE:
        s1, a, s2, isterminal, r = memory.get_sample(BATCH_SIZE)

        # this is original implementation
        q = get_q_values(s2).cpu().data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).cpu().data.numpy()
        #this is double q netwrok (chen) implementation
        #q2 = get_double_network_q_values(s1)


        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + DISCOUNT_FACTOR * (1 - isterminal) * q2
        learn(s1, target_q)

def generate_actions(n, act_type):
    if act_type == "single":
       actions = []
       for i in range(n):
            current_action = [False] * n
            current_action[i] = True
            actions.append(current_action)
       return actions
    elif act_type == "multy":
        return [list(a) for a in it.product([0, 1], repeat=n)]



def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""
    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        # const_eps_EPOCHS = 0.1 * EPOCHS  # 10% of learning time
        # eps_decay_EPOCHS = 0.6 * EPOCHS  # 60% of learning time

        const_eps_EPOCHS = EPS_DECAY_10 * EPOCHS  # 10% of learning time
        eps_decay_EPOCHS = EPS_DECAY_60 * EPOCHS  # 60% of learning time

        if epoch < const_eps_EPOCHS:
            return start_eps
        elif epoch < eps_decay_EPOCHS:
            # Linear decay
            return start_eps - (epoch - const_eps_EPOCHS) / \
                   (eps_decay_EPOCHS - const_eps_EPOCHS) * (start_eps - end_eps)
        else:
            return end_eps


    s1 = we.game_state_to_image(game.get_state(),game).T


    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    s1 = s1.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
    if random() <= eps:
        a = randint(0, len(actions) - 1)
        q = get_q_values(s1)
    else:
        # Choose the best action according to the network.
        a,q = get_best_action(s1)
    if PRINT_TRAINING_PROCESS:
        state_string = sp.calc_state_string(game.get_state(), game)
        print("Game state: ", state_string)
        print("Q-Values: ", q.data)
        print("Chosen action: ", actions[a])
    reward = game.make_action(actions[a], FRAME_REPEAT)

    isterminal = game.is_episode_finished()
	#s2 might be a bug, we need to save the state of the parser
#    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
    s2 = we.game_state_to_image(game.get_state(),game).T.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]]) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()
    return 1 #interactions this step


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(CONFIG_FILE_PATH,seed):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(CONFIG_FILE_PATH)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_seed(seed)
    game.init()
    print("Doom initialized.")
    return game


def seed_torch(seed=1029):
    rnd.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Create Doom instance
    globals().update(parse_arguments())
    seed_torch(SEED)
    RESOLUTION = (SENTANCE_LEN, WORD_EMBEDDING_LEN)
    # Configuration file path
    MODEL_SAVEFILE = "./models/model-doom_nlp_" + SCENARIO + "_seed_" + str(SEED) + ".pth"
    CONFIG_FILE_PATH = "scenarios/" + SCENARIO + ".cfg"

    gpu_id = str(get_free_gpu())
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    print("chosen gpu: " + str(device))

    criterion = nn.MSELoss()
    if N_PATCHES == 3:
        sp = StoryParser(1,REVERSE_GREEN)
    else:
        sp = PatchStoryParser(1,N_PATCHES,3,REVERSE_GREEN)
    we = GloveEmbedder(sp, WORD_EMBEDDING_LEN, SENTANCE_LEN)
    pg = PlotGenerator("dqn_green_",SCENARIO,ARCH,False,FRAME_REPEAT,str(SEED),N_PATCHES)
    game = initialize_vizdoom(CONFIG_FILE_PATH,SEED)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = generate_actions(n,ACTION_TYPE)

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=REPLAY_MEMORY_SIZE,rep_type=REP_TYPE,resolution=RESOLUTION)

    if LOAD_MODEL:
        print("Loading model from: ", MODEL_SAVEFILE)
        model = torch.load(MODEL_SAVEFILE)
        target_model = model
    else:
        model = TextCNN(len(actions),WORD_EMBEDDING_LEN,SENTANCE_LEN,HIDDEN_UNITS,FILTER_COUNT,device)
        target_model = TextCNN(len(actions),WORD_EMBEDDING_LEN,SENTANCE_LEN,HIDDEN_UNITS,FILTER_COUNT,device)
        target_model.load_state_dict(model.state_dict())
    model.cuda(device)
    target_model.cuda(device)
    #optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    total_train_episodes = 0
    total_interactions = 0

    print("Starting the training!")
    time_start = time()
    if not SKIP_LEARNING:
        for epoch in range(EPOCHS):
            print("\n Scenario: ", SCENARIO)
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
                total_interactions += perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
            total_train_episodes += train_episodes_finished

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            if len(train_scores) > 0:
                print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(TEST_EPISODES_PER_EPOCH, leave=False):
                game.new_episode()
                while not game.is_episode_finished():

                    s1 = we.game_state_to_image(game.get_state(), game).T
                    state = s1.reshape([1,1,RESOLUTION[0], RESOLUTION[1]])
                    best_action_index,_ = get_best_action(state)

                    #game.make_action(actions[best_action_index], FRAME_REPEAT)
                    game.make_action(actions[best_action_index],1)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", MODEL_SAVEFILE)
            torch.save(model, MODEL_SAVEFILE)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

            #update plot generator
            pg.update_reward_data(train_scores.mean(),test_scores.mean())
            pg.update_episodes_data(total_train_episodes)
            pg.update_interactions_data(total_interactions)

            #update target network
            target_model.load_state_dict(model.state_dict())


    game.close()

    # if not SKIP_LEARNING:
    #     pg.plot_reward_progress()
    #     pg.dump_data()


    # Reinitialize the game with window visible
    print(SKIP_TEST)
    if SKIP_TEST == True:
        print("======================================")
        print("Training finished. It's time to watch!")
        input("Press any key to continue...")
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        for _ in range(EPISODES_TO_WATCH):
           game.new_episode()
           while not game.is_episode_finished():
               #state = preprocess(game.get_state().screen_buffer)
               s1 = we.game_state_to_image(game.get_state(), game).T
               state = s1.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
               best_action_index,q = get_best_action(state)

               state_string = sp.calc_state_string(game.get_state(), game)
               print("Game state: ", state_string)
               print("Q-Values: ",q.data)
               print("Chosen action: ", actions[best_action_index])


               # Instead of make_action(a, FRAME_REPEAT) in order to make the animation smooth
               game.make_action(actions[best_action_index], 1)
               # game.set_action(actions[best_action_index])
               # for _ in range(1):
               #     game.advance_action()

               #game.make_action(actions[best_action_index],FRAME_REPEAT)

           # Sleep between episodes
           sleep(1.0)
           score = game.get_total_reward()
           print("Total score: ", score)
