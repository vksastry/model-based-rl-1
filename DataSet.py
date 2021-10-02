"""Supplies data for the AT model."""

import pandas as pd
#from sklearn.decomposition import PCA
import glob
import random





#import os
from typing import Tuple
import torch
#import torch.nn as nn
#import torchvision

from torch.utils.data import TensorDataset, DataLoader

#from tqdm import tqdm
import numpy as np
#from skimage.transform import resize
#from sklearn.utils import shuffle






def prepare_dataloader() -> Tuple[torch.utils.data.DataLoader]:
    # Prepare DataLoader
    path = '../data/AT/XTotal1.at.gz'
    pathY = path.replace(".at", ".price.at")

    npX = np.load(path)
    npY = np.load(pathY)


    #download and load training data
    BATCH_SIZE = 20
    trainloader = DataLoader(npX,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=False,
                             num_workers=4)
    validloader = DataLoader(npY,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=False,
                             num_workers=4)
    return trainloader, validloader


class DataSet:
    """Supplies data for the AT model."""

    def __init__(self, maxMoves, useNormalGain=True):
        """Initialize the data set."""
        self.train_loader, self.test_loader = prepare_dataloader()
        self.trainSize = None
        self.testSize = None


    def getFeaturesAndPrices(self, train):
        """Get feature data."""
        if train:


            # Is this a batch?  Yes.  Does the old code do a batch?  Yes.
            (self.train_features, self.dfYTrain) = next(self.train_loader)
            self.trainSize = len(self.train_features)

            # TODO: Convert to DataFrames

            return self.train_features, self.dfYTrain
        else:


            # Is this a batch?  Yes.  Does the old code do a batch?  Yes.
            (self.test_features, self.dfYTest)  = next(self.test_loader)
            self.testSize = len(self.test_features)

            # TODO: Convert to DataFrames

            return self.test_features, self.dfYTest

    def getSize(self, train):
        """Get size of the data."""
        if train:
            return self.trainSize
        else:
            return self.testSize

