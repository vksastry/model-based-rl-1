from types import SimpleNamespace
import numpy as np

import DataSet





import os
from typing import Tuple
import torch
#import torch.nn as nn
#import torchvision


"""
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from sklearn.utils import shuffle
import pandas as pd
"""






#globals

g_nStocks         = 1
g_samplesPerEvent = 5
g_eventsPerDay    = 13
g_daysOfHistory   = 22
g_samples = g_samplesPerEvent * g_eventsPerDay * g_daysOfHistory

g_maxDaysToPlay   = 30
g_maxMoves        = g_eventsPerDay * g_maxDaysToPlay

g_nFeatures       = g_samples

g_action_space = list( range(2) )


class ATEnv:
    """AT Environment."""

    def __init__(self):
        """Initialize the AT environment."""
        print("Overriding random starting index into each set of data.")
        self.reset(withObservation=False)

        self._elapsed_steps = 0

        self.action_space = SimpleNamespace(**{'n':len(g_action_space)})

        numChannels = 1
        self.observation_space = np.zeros((numChannels, g_nStocks, g_nFeatures))

    def seed(self, seed):
      return

    def reset(self, withObservation=True):
        """Reset the board to its initial state."""
        """ ****** Not tested. """
        self.useNormalGain = True
        self.data = DataSet.DataSet(g_maxMoves, self.useNormalGain)

        # There is only one stock at a time being played.
        # Call AT to generate the list of closing prices for each stock
        training = True
        self.closes = self.data.getPrices(training)

        # Get the features as a list of one DF per stock make sure the order is the same
        self.featuresDf = self.data.getFeatures(training)

        self.max = self.data.getSize(training) - 1

        #if self.max < g_maxMoves - 1:
        #    print(f'self.max: {self.max}; g_maxMoves: {g_maxMoves}')
        #self.max = g_maxMoves

        # step immediately increments time so use -1.
        self.time = -1

        # 0 represents we own no stock.
        self.last_action = 0

        self._elapsed_steps = 0

        if withObservation:
            return self.get_observation()


    def legal_actions(self):
        """Return legal actions."""
        """ Tested """
        return g_action_space

    def step(self, action):
        """Take one step."""
        self.last_action = action
        self.time += 1

        observation = self.get_observation()
        reward      = self.getReward()
        done        = self.time >= self.max

        #if done:
        #    print(f'self.time: {self.time}; self.max: {self.max}')

        self._elapsed_steps += 1
        info = {}

        return observation, reward, done, info

    def getRewardNormalGain(self):  # sourcery skip: assign-if-exp
        """Get the reward for the last move."""
        """Gains need to be multiplied in MuZero.py."""




        """Gains need to be multiplied in MuZero.py."""




        """ Not Tested """
        if self.last_action == 0:
            haveStock = False
        else:
            haveStock = True

        #historicClose = self.closes.loc[self.time, 'historic_close']
        #futureClose   = self.closes.loc[self.time, 'future_close']
        #gain          = self.closes.loc[self.time, 'gain']
        score         = self.closes.loc[self.time, 'score']

        if haveStock:
            # A gain is a gain.  A loss is a loss.
            reward = score
        else:
            if score > 1.0:
                # We do not have the stock so, a gain becomes a loss.
                # Turn it into a loss.
                reward = 1 - (score - 1.0)
            elif score < 1.0:
                # We do not have the stock so, a loss becomes a gain.
                # Turn it into a gain.
                reward = 1 + (1.0 - score)
            else:
                # Score is one.
                reward = score
        reward = (reward - 1.0) * 100000.0
        #print(f"reward: {reward}", end="\r",)

        return reward



    # TODO:  Try to scale gain between +/-1


    def getRewardAdjustedGain(self):
        # sourcery skip: assign-if-exp, inline-immediately-returned-variable
        """Get the reward for the last move."""
        """ Not Tested """
        if self.last_action == 0:
            multiplier = -1
        else:
            multiplier = 1

        score         = self.closes.loc[self.time, 'score']
        reward = score * multiplier

        return reward


    def getReward(self):  # sourcery skip: lift-return-into-if
        """Get the reward for the last move."""
        """ Not Tested """
        if self.useNormalGain:
            reward = self.getRewardNormalGain()
        else:
            reward = self.getRewardAdjustedGain()

        return reward


    def test_active(self, status):
        """What does this do?."""
        if status:
            training        = False
            self.closes     = self.data.getPrices(training)
            self.featuresDf = self.data.getFeatures(training)
            self.max        = self.data.getSize(training) - 1
            self.reset(withObservation=False)

    def render(self, mode=''):
        """Render the current state of the game."""
        """ Tested """
        reward       = self.getReward()
        legalActions = self.legal_actions()

        print(f"reward: {reward}")
        print(f"legalActions: {legalActions}")

        # Used for unit testing.
        return { "reward": reward, "legalActions": legalActions }

    def get_observation(self):
        # sourcery skip: inline-immediately-returned-variable
        """Get the current observation."""
        """ Tested """

        # TODO: Better test it again.  It is returning 21 rows.
        #print(self.featuresDf)

        self.time = max(self.time, 0)

        featureRowAsPandasArray = self.featuresDf.iloc[self.time]
        # observation must have three dimensions.
        featureRowAsList = featureRowAsPandasArray.tolist()
        observation = [[featureRowAsList]]

        return observation
