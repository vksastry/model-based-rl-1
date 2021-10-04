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



def getData(pathX):
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

    # *** Convert to numpy array.
    npX = dfX.to_numpy()
    npY = dfY.to_numpy()


    train_data = []
    for i in range(len(npX)):
        train_data.append([npX[i], npY[i]])

    return train_data


def prepare_dataloader() -> Tuple[torch.utils.data.DataLoader]:
    # Prepare DataLoader
    pathX = './data/AT/XTotal2A.at'
    train_data = getData(pathX)

    #trainloader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=20)
    #i1, l1 = next(iter(trainloader))
    #print(i1.shape)
    #print(l1.shape)
    #input('Press <Enter> to continue...')

    pathX = './data/AT/XTotal3A.at'
    test_data = getData(pathX)




    #download and load training data
    BATCH_SIZE = 400
    trainloader = DataLoader(train_data,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=False,
                             num_workers=1)

    #
    # TODO: This dataset has to be changed.
    #
    validloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=False,
                             num_workers=1)
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
            xTrain, yTrain = next(iter(self.train_loader))
            self.trainSize = len(xTrain)

            # Convert to DataFrames
            dfXTrain = pd.DataFrame(xTrain)
            dfYTrain = pd.DataFrame(yTrain, columns=columns)
            #print('dfYTrain.head(): ', dfYTrain.head())
            #input('Press <Enter> to continue')
            return dfXTrain, dfYTrain
        else:
            xTest, yTest = next(iter(self.test_loader))
            self.testSize = len(xTest)

            # Convert to DataFrames
            dfXTest = pd.DataFrame(xTest)
            dfYTest = pd.DataFrame(yTest, columns=columns)
            #print('dfYTest.head(): ', dfYTest.head())
            #input('Press <Enter> to continue')
            return dfXTest, dfYTest

    def getSize(self, train):
        """Get size of the data."""
        if train:
            return self.trainSize
        else:
            return self.testSize

