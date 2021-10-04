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

    npX = dfX.to_numpy()
    npY = dfY.to_numpy()

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
        self.useNormalGain = useNormalGain
        self.split = 0.8
        dfFeatures = self.collectData(maxMoves)
        self.splitData(dfFeatures, maxMoves)


    def collectData(self, maxMoves, pattern='data/XTotal*.at'):
        """Collect data."""
        files = [file for file in glob.glob(pattern)]

        # Keep files that do not contain '.price.'.
        filesShortList = [file for file in files if file.find('.price.') == -1]
        # Generate random integer between zero and len(files)
        randomIndex = random.randint(0, len(filesShortList) - 1)

        path = filesShortList[randomIndex]
        pathY = path.replace(".at", ".price.at")

        self.dfY = pd.read_csv(pathY, header=None, names=['historic_close', 'future_close'])

        totalNumRows = len(self.dfY.index)
        # Get enough for testing.
        numRowsRequired = maxMoves * (1 + (1 - self.split))
        numRowsRequired = int(numRowsRequired)

        # Create a random starting point so that we don't always start at zero.
        startIndex = random.randint( 0, totalNumRows - numRowsRequired - 1 )
        startIndex = 0
        startIndex = random.randint( 0, 60 )

        dfFeatures = pd.read_csv(path, header=None, skiprows=startIndex, nrows=maxMoves)

        self.dfY = pd.read_csv(pathY, header=None, skiprows=startIndex, nrows=maxMoves, names=['historic_close', 'future_close'])
        self.dfY['gain'] = self.dfY['future_close'] / self.dfY['historic_close']

        if self.useNormalGain:
            self.dfY['score'] = self.dfY['gain']
        else:
            # Subracting 1 from 'gain' to make losses negative.
            self.dfY['score'] = self.dfY['gain'].sub(1)

        return dfFeatures


    def splitData(self, dfFeatures, maxMoves):
        """Establish the data."""
        self.dataSize = len(dfFeatures.index)
        #print(f'self.dataSize: {self.dataSize}')

        self.trainSize = maxMoves
        self.testSize  = self.dataSize - self.trainSize

        self.train_features = dfFeatures[:self.trainSize]
        if len(self.train_features.index) != self.trainSize:
            print(f'{len(self.train_features)} != {self.trainSize}')
            input('train_features size mismatch with self.trainSize.  Press <Enter> to continue')

        self.test_features = dfFeatures[self.trainSize:]
        if len(self.test_features) != self.testSize:
            print(f'{len(self.test_features)} != {self.testSize}')
            input('test_features size mismatch with self.testSize. Press <Enter> to continue')


    def getFeatures(self, train):
        """Get feature data."""
        if train:
            return self.train_features
        else:
            return self.test_features

    def getPrices(self, train):
        """Get price data."""
        if train:
            return self.dfY.head(self.trainSize)
        else:
            return self.dfY.tail(self.test_features)

    def getSize(self, train):
        """Get size of the data."""
        if train:
            return self.trainSize
        else:
            return self.test_features

