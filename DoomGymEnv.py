import gym
from gym import spaces
from vizdoom import *
import numpy as np
import os
#from gym.envs.classic_control import rendering
from AnacdoteParser import AnacdoteParser
#from StoryParser import StoryParser
from ManyPatchStoryParser import StoryParser as PatchStoryParser
from NaturalParser import NaturalParser as StoryParser
from VectorParser import VectorParser
from GloveEmbedder import GloveEmbedder
from LogicSegmentMap import LogicSegmentMap
import skimage

CONFIGS = [['basic.cfg', 3],                        # 0
           ['take_cover.cfg', 2],                   # 1
           ['defend_the_center.cfg', 3],            # 2
           ['defend_the_line.cfg', 3],              # 3
           ['health_gathering.cfg', 3],             # 4
           ['basic_middle.cfg', 3],                 # 5
           ['take_cover_middle.cfg', 2],            # 6
           ['defend_the_center_middle.cfg', 3],     # 7
           ['defend_the_line_middle.cfg', 3],       # 8
           ['health_gathering_middle.cfg', 3],      # 9
           ['basic_extreme.cfg', 3],                # 10
           ['take_cover_extreme.cfg', 2],           # 11
           ['defend_the_center_extreme.cfg', 3],    # 12
           ['defend_the_line_extreme.cfg', 3],      # 13
           ['health_gathering_extreme.cfg', 3],     # 14
           ['super_scenario.cfg',7]]                # 15

# Converts and down-samples the input image
def preprocess(img,n_channels,res_0,res_1):
    img = skimage.transform.resize(img, (n_channels,res_0,res_1))
    img = img.astype(np.float32)
    return img


def scenario_to_level(scenario):
    for i,sce_duo in enumerate(CONFIGS):
        if scenario + ".cfg" == sce_duo[0]:
            return i


class VizdoomEnv(gym.Env):

    def __init__(self,env_id,scenario,seed,rep_type,resolution,n_channels,patch_count=3,reverse_green=False):

        # init game
        self.env_id = env_id
        self.rep_type = rep_type
        self.resolution = resolution
        self.n_channels = n_channels
        self.game = DoomGame()
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[scenario_to_level(scenario)][0]))
        self.game.set_labels_buffer_enabled(True)
        self.game.set_window_visible(False)
        self.game.set_seed(seed)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        print("Doom initialized.")
        self.state = None
        self.action_space = spaces.Discrete(CONFIGS[scenario_to_level(scenario)][1])
        self.patch_count = patch_count
        self.reverse_green = reverse_green

        #define the observarion_space depending on the rep_type, ran,nlp,vec,seg,viz
        if rep_type in ["seg","viz"]:
            self.observation_space = spaces.Box(0, 255, (n_channels,
                                                         resolution[0],
                                                         resolution[1],
                                                         ),dtype=np.uint8)
            self.lsm = LogicSegmentMap(self.n_channels,(640,480))
        elif rep_type == "nlp": #need to init parser and embedder
            self.observation_space = spaces.Box(-5,5, (  resolution[0],
                                                         resolution[1]), dtype=np.float64)
            if self.patch_count == 3:
                self.sp = StoryParser(1)
            else:
                self.sp = PatchStoryParser(1,self.patch_count,3,green_monster=reverse_green)
            self.we = GloveEmbedder(self.sp,resolution[1],resolution[0])
        else: # vec
            self.observation_space = spaces.Box(0,255,(90,), dtype=np.uint8)
            self.vp = VectorParser(1)

        self.viewer = None

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        if action >= len(act):
            print("wait")
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            if self.rep_type == "viz":
                #observation = np.transpose(state.screen_buffer, (1, 2, 0)) #need func from dqn_vision
                observation = preprocess(state.screen_buffer,self.n_channels,self.resolution[0],self.resolution[1])
            elif self.rep_type == "seg":
                observation = self.lsm.state_to_logic_map(state)
                observation = preprocess(observation, self.n_channels, self.resolution[0], self.resolution[1])
            elif self.rep_type == "nlp":
                #if "right" in self.sp.calc_state_string(state,self.game):
                    #print("here")
                observation = self.we.game_state_to_image(state,self.game).T #fixed here
                observation = observation.reshape([1,1,self.resolution[0],self.resolution[1]])
            else: #vec
                observation = self.vp.calc_state_string(state,self.game)
                observation = observation.reshape([1,self.resolution[0],self.resolution[1]])
        else:
            # observation = np.uint8(np.zeros(self.observation_space.shape))
            observation = np.zeros(self.observation_space.shape)

        info = {'dummy': 0}

        return observation, reward, done, info

    def reset(self):
        self.game.new_episode()
        if self.rep_type == "nlp":
            self.state = self.we.game_state_to_image(self.game.get_state(),self.game).T
            return self.state.reshape([1,200,50])
        elif self.rep_type == "viz":
            self.state = preprocess(self.game.get_state().screen_buffer,self.n_channels,self.resolution[0],self.resolution[1])
            return self.state
        elif self.rep_type == "seg":
            observation = self.lsm.state_to_logic_map(self.game.get_state())
            self.state = preprocess(observation, self.n_channels, self.resolution[0], self.resolution[1])
            return self.state
        elif self.rep_type == "vec":
            observation = self.vp.calc_state_string(self.game.get_state(), self.game)
            observation = observation.reshape([1, self.resolution[0], self.resolution[1]])
            self.state = observation
            return self.state

        raise NotImplementedError

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
        except AttributeError:
            pass

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys


