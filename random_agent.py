#!/usr/bin/env python

from __future__ import print_function
from random import choice
from vizdoom import DoomGame,GameVariable,ScreenResolution,Mode,ScreenFormat
from tqdm import trange
import numpy as np
from ArgumentParser import parse_arguments
import itertools as it
from PlotGenerator import PlotGenerator


TEST_EPISODES_PER_EPOCH = 10


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(CONFIG_FILE_PATH,seed):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(CONFIG_FILE_PATH)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_seed(seed)
    game.init()
    print("Doom initialized.")
    return game


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

if __name__ == "__main__":
    globals().update(parse_arguments())
    CONFIG_FILE_PATH = "scenarios/" + SCENARIO + ".cfg"
    game = initialize_vizdoom(CONFIG_FILE_PATH,SEED)
    n = game.get_available_buttons_size()
    actions = generate_actions(n,ACTION_TYPE)
    pg = PlotGenerator(SCENARIO,ARCH,False,FRAME_REPEAT,str(SEED))


    # Sleep time between actions in ms
    sleep_time = 28


    for i in range(EPOCHS):
        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        print("\nTesting " + SCENARIO + "... epoch #" + str(i))
        total_interactions = i*LEARNING_STEPS_PER_EPOCH
        test_episode = []
        test_scores = []
        for test_episode in trange(TEST_EPISODES_PER_EPOCH, leave=False):
            game.new_episode()
            while not game.is_episode_finished():
                game.make_action(choice(actions), FRAME_REPEAT)

            r = game.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f +/- %.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())
        pg.update_reward_data(None,test_scores.mean())
        pg.update_interactions_data(total_interactions)
        print("Episode finished!")
        print("************************")
    pg.dump_data()
    pg.plot_reward_progress()

