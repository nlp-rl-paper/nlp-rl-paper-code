from Parser import Parser
from vizdoom.vizdoom import GameState
import numpy as np
import skimage.transform


class LogicSegmentMap(Parser):
    def __init__(self,n_layers,resolution,hist_len=1):
        super(LogicSegmentMap, self).__init__(hist_len)
        self.n_layers = n_layers
        self.resolution = resolution


    def state_to_logic_map(self,game_state : GameState):
        seg_map = np.zeros([self.resolution[0],self.resolution[1]])
        #label_buffer = self._preprocess(game_state.labels_buffer)
        label_buffer = game_state.labels_buffer
        for label_index,label in enumerate(game_state.labels):
            if label.object_name in self.enemies_melee:
                seg_layer = 0
            elif label.object_name in self.enemies_ranged:
                seg_layer = 1
            elif label.object_name in self.ammo_packages:
                seg_layer = 2
            elif label.object_name in self.armors:
                seg_layer = 3
            elif label.object_name in self.weapons:
                seg_layer = 4
            elif label.object_name in self.medkits:
                seg_layer = 5
            elif label.object_name in self.obstacles:
                seg_layer = 6
            elif label.object_name == "DoomImpBall":
                seg_layer = 7
            else:
                seg_layer = -1 #not reachable
            label_value = label.value
            label_indexes = np.array(np.argwhere(label_buffer == label_value)).T
            seg_map[label_indexes[1],label_indexes[0]] = seg_layer
        return seg_map

