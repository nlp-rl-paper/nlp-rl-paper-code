from vizdoom.vizdoom import GameVariable,GameState
import numpy as np
import random
from textblob import Word
from StoryParser import StoryParser
import nltk

HEALTH_LOW_RATE = 10
AMMO_LOW_RATE = 5
ENEMIES_HIGH_RATE = 2
CENTER_ANGLE_RANGE = 20
RIGHT_ANGLE_RANGE = 60
SCREEN_WIDTH_CENTER = 320
SCREEN_HEIGHT_CENTER = 240
SCREEN_WIDTH = 2*SCREEN_WIDTH_CENTER
BLOOD_HIT_PLAYER_THRESHOLD = 100
PATCHES_COUNT = 3
CLOSE_TO_PLAYER_THRESHOLD = 200
FAR_FROM_PLAYER_THRESHOLD = 300
DOOM_IMPBALL_CLOSE = 200
DOOM_IMPBALL_FAR = 350

class BlobParser(StoryParser):
    def __init__(self,hist_size,reverse_green=False):
        super(BlobParser,self).__init__(hist_size)
        #nltk.download("wordnet")


    def make_blob(self, yours, swap_rate):
        mine = []
        for string_word in yours:
            word_object = Word(string_word)
            if random.randint(0, swap_rate - 1) == 0:
                meaning_count = len(word_object.synsets)
                if meaning_count > 0:
                    meaning_selected = random.randint(0, meaning_count - 1)
                    lemmas = word_object.synsets[meaning_selected].lemmas()
                    synonym_count = len(lemmas)
                    mine += [lemmas[random.randint(0, synonym_count - 1)].name()]
                else:
                    mine += [string_word]
            else:
                mine += [string_word]

        return ''.join(mine)

    def calc_state_string(self,cur_game_state : GameState ,game_variable : GameVariable):
        state_str = super(BlobParser,self).calc_state_string(cur_game_state,game_variable)
        return self.make_blob(state_str,10)


