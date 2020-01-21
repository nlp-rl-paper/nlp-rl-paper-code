from Parser import Parser
from StoryParser import StoryParser
from ManyPatchStoryParser import StoryParser as ManyPatchStoryParser
from AnacdoteParser import AnacdoteParser
from SimpleParser import SimpleParser
from RoyParser import RoyParser
from BlobParser import BlobParser
from vizdoom.vizdoom import GameVariable,GameState
from random import randint,choice
from inspect import stack



class NaturalParser(Parser):
    def __init__(self,hist_size=1,
                 reverse_green=False,
                 chosen_parser=-1):
        super(NaturalParser, self).__init__(hist_size)

        self._parser_list = []
        self._parser_list.append(StoryParser(hist_size,False))
        self._parser_list.append(ManyPatchStoryParser(hist_size,5,3,False))
        self._parser_list.append(SimpleParser(hist_size))
        self._parser_list.append(AnacdoteParser(hist_size))
        self._parser_list.append(RoyParser(hist_size))
        self._parser_list.append(BlobParser(hist_size))
        # add each parser and increment _num_parsers
        self._num_parsers = 6
        self._chosen_parser = chosen_parser


    def _natural_parser_decorator (self, * args, ** kwargs): #implement using a decorator
        chosen_parser = choice(self._parser_list)
        return getattr(chosen_parser,stack()[1][3])(*args,**kwargs)


    def calc_state_string(self,game_state,game_variable):
        return self._natural_parser_decorator(game_state,game_variable)


    def parse_state(self,game_state,game_variable):
        return self._natural_parser_decorator(game_state,game_variable)









