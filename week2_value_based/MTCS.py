import gym
import numpy as np
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple

ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))
# Wrappers are used to transform an environment in a moduler way (gym docs's definition)
class WithSnapshots(Wrapper):
    def get_snapshot(self):
        self.render()
        if self.unwrapped.viewer is not None:
            self.unwrapped.viewer.close()
            self.unwrapped.viewer = None
        return dumps(self.env)
    
    def load_snapshot(self, snapshot):
        assert not hasattr(self, "_monitor") or  hasattr(self.env, "_monitor"), "can't backtrack while recording"
        self.render()
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        self.reset()
        load_snapshot(snapshot)
        observation, reward, done, info = self.step(action)
        return ActionResult(snapshot, observation, reward, done, info)

