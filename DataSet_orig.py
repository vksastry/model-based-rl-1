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
    pathX = './data/AT/XTotal2.at'
    pathY = pathX.replace(".at", ".price.at")

    dfX = pd.read_csv(pathX)
    dfY = pd.read_csv(pathY, header=None, names=['historic_close', 'future_close'])
    dfY['gain'] = dfY['future_close'] / dfY['historic_close']

    useNormalGain = True
    if useNormalGain:
        dfY['score'] = dfY['gain']
    else:
        # Subracting 1 from 'gain' to make losses negative.
        dfY['score'] = dfY['gain'].sub(1)


    #download and load training data
    BATCH_SIZE = 20
    trainloader = DataLoader(dfX,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=False,
                             num_workers=4)
    validloader = DataLoader(dfY,
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
        columns = ['historic_close', 'future_close', 'gain', 'score']

        if train:
            # Is this a batch?  Yes.  Does the old code do a batch?  Yes.
            (xTrain, yTrain) = next(self.train_loader)
            self.trainSize = len(xTrain)

            # Convert to DataFrames
            dfXTrain = pd.DataFrame(xTrain)
            dfYTrain = pd.DataFrame(yTrain, columns=columns)
            print('dfYTrain.head(): ', dfYTrain.head())
            input('Press <Enter> to continue')
            return dfXTrain, dfYTrain
        else:
            # Is this a batch?  Yes.  Does the old code do a batch?  Yes.
            (xTest, yTest)  = next(self.test_loader)
            self.testSize = len(xTest)

            # TODO: Convert to DataFrames
            dfXTest = pd.DataFrame(xTest)
            dfYTest = pd.DataFrame(yTest, columns=columns)
            return dfXTest, dfYTest

    def getSize(self, train):
        """Get size of the data."""
        if train:
            return self.trainSize
        else:
            return self.testSize

