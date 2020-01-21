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
from random import randint
import numpy as np
import cv2

if __name__ == "__main__":
    game = DoomGame()

    # Use other config file if you wish.
    game.load_config("scenarios/defend_the_center.cfg")
    #game.load_config("scenarios/take_cover.cfg")
    
    game.set_render_hud(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)

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

    # Sleep time between actions in ms
    sleep_time = 28

    #sp = AnacdoteParser(1)
    sp = MultiPatchStoryParser(1,3,3,green_monster=True)
    vp = VectorParser(1)
    we = GloveEmbedder(sp,50,200)
    lsm = LogicSegmentMap(6,(640,480))

    flag = False
    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():
            # Gets the state and possibly to something with it

            for i in range(episodes):
                # Not needed for the first episode but the loop is nicer.
                game.new_episode()
                while not game.is_episode_finished():
                    state = game.get_state()
                    s = sp.parse_state(state, game)[0]
                    print(s)
                    current_count = len(s.split(" "))
                    game.make_action(actions[randint(0, n_actions - 1)], 4)
            print(sp.parse_state(state,game)[0])
            # print("State:")
            # print(sp.parse_state(state, game)[0])
            #print(len(sp.parse_state(state, game)[0].split(" ")))

            # if flag == False:
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # fig,ax = plt.subplots(nrows=1,ncols=2)
                #ax.set_aspect('auto')
                # plt.title("game state GloVe embedding")
                # ax.imshow(we.game_state_to_image(state,game))
                # ax.set_aspect('auto')
                # plt.xlabel("word")
                # plt.ylabel("embedding")
                # plt.show(block=False)
                #print(len(vp.parse_state(state, game)[0]))

                #seg_map = lsm.state_to_logic_map(state).T
                #plt.figure()
                #plt.title("image segmentation map")
                # raw_img = np.swapaxes(state.screen_buffer,1,2).T
                # ax[0].imshow(raw_img)
                # print(raw_img.shape)
                # print(raw_img.T.shape)
                # ax[0].axis("off")
                #
                # ax[1].imshow(seg_map)
                # ax[1].axis("off")
                # plt.show()

                # flag = True
            # for i,label in enumerate(state.labels):
            #     print("distance: " + str(sp.calc_distance_string(state,i)))


            # action = int(input("")) #0 - left, 1 - right, 2 - shoot
            # game.make_action(actions[action],4)
            print()


        print("Episode finished!")
        print("************************")

    cv2.destroyAllWindows()