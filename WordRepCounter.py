#!/usr/bin/env python

from __future__ import print_function
from random import choice
from vizdoom import DoomGame,GameVariable,ScreenResolution
import matplotlib.pyplot as plt
from SimpleParser import SimpleParser
from WordEmbedder import WordEmbedder
from GloveEmbedder import GloveEmbedder
from dqn_nlp import generate_actions
from AnacdoteParser import AnacdoteParser
from ToVectorParser import ToVectorParser
from StoryParser import StoryParser
from ManyPatchStoryParser import StoryParser as MultiPatchStoryParser
from VectorParser import VectorParser
from LogicSegmentMap import LogicSegmentMap
import numpy as np
from random import randint
import cv2

if __name__ == "__main__":
    patches = [31,5,11,21]
    scenarios = ["basic"]
                 # "basic_middle",
                 # "basic_extreme",
                 # "defend_the_center",
                 # "defend_the_center_middle",
                 # "defend_the_center_extreme"]
    for patches_count in patches:
        for scenario in scenarios:
            game = DoomGame()

            # Use other config file if you wish.

            game.load_config("scenarios/" + scenario + ".cfg")
            game.set_render_hud(True)
            game.set_screen_resolution(ScreenResolution.RES_640X480)
            game.set_window_visible(False)
            # Enables labeling of the in game objects.
            game.set_labels_buffer_enabled(True)

            game.clear_available_game_variables()
            game.add_available_game_variable(GameVariable.POSITION_X)
            game.add_available_game_variable(GameVariable.POSITION_Y)
            game.add_available_game_variable(GameVariable.POSITION_Z)
            game.add_available_game_variable(GameVariable.CAMERA_POSITION_X)
            game.add_available_game_variable(GameVariable.CAMERA_POSITION_Y)

            game.init()

            actions = generate_actions(game.get_available_buttons_size(),"single")
            n_actions = len(actions)

            episodes = 10
            counts = []
            current_count = None
            # Sleep time between actions in ms
            sleep_time = 28

            #sp = AnacdoteParser(1)
            sp = MultiPatchStoryParser(1,patches_count,3)
            #sp = StoryParser(1)


            flag = False
            for i in range(episodes):
                # Not needed for the first episode but the loop is nicer.
                game.new_episode()
                while not game.is_episode_finished():
                    s = sp.parse_state(game.get_state(), game)[0]
                    print(s)
                    current_count = len(s.split(" "))
                    game.make_action(actions[randint(0,n_actions-1)],4)
                counts.append(current_count)


            count_mean = np.mean(counts)
            count_std  = np.std(counts)

            print("scenario: " + scenario + " patches: " + str(patches_count))
            print("mean: " + str(count_mean))
            print("std: " + str(count_std))

            cv2.destroyAllWindows()