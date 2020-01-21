import numpy as np
import random

class ReplayMemory:
    def __init__(self, capacity,rep_type,resolution,n_channels=1,seed=0):
        random.seed(seed)
        if rep_type == "vec":
            state_shape = (capacity, resolution[0], resolution[1])
        else:
            if rep_type == "viz":
                channels = n_channels
            elif rep_type == "nlp":
                channels = 1
            else:
                channels =n_channels
            state_shape = (capacity, channels ,resolution[0], resolution[1])
        self.rep = rep_type
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        if self.rep == "vec":
            self.s1[self.pos, :, :] = s1
        else:
            self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            if self.rep == "vec":
                self.s2[self.pos, :, :] = s2
            else:
                self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = random.sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]