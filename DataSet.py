"""Supplies data for the AT model."""

import pandas as pd
#from sklearn.decomposition import PCA
import glob
import random

class DataSet:
    """Supplies data for the AT model."""

    def __init__(self, maxMoves, useNormalGain=True):
        """Initialize the data set."""
        #print('__init__.maxMoves: ', maxMoves)

        self.useNormalGain = useNormalGain
        self.split = 0.8
        dfFeatures = self.collectData(maxMoves)


    def collectData(self, maxMoves, pattern='data/AT/XTotal*.at'):
        """Collect data."""
        #
        # Find files and size of data.
        #
        files = [file for file in glob.glob(pattern)]

        # Keep files that do not contain '.price.'.
        filesShortList = [file for file in files if file.find('.price.') == -1]
        # Generate random integer between zero and len(files)
        randomIndex = random.randint(0, len(filesShortList) - 1)

        path = filesShortList[randomIndex]
        pathY = path.replace(".at", ".price.at")

        # Use the Y data to get the total number of rows because the file is so 
        # much smaller than features.
        self.dfTempY = pd.read_csv(pathY, header=None)

        totalNumRows = len(self.dfTempY.index)
        #print(f'totalNumRows: {totalNumRows}')



        #
        # Select training data only from the first part of the data split.
        #
        totalRowsInTrainingArea = int(self.split * totalNumRows)
        totalRowsInTestingArea  = totalNumRows - totalRowsInTrainingArea

        self.trainSize = maxMoves
        # Get enough for testing.
        #print(f'maxMoves: {maxMoves}')
        numRowsRequired = maxMoves
        #print(f'numRowsRequired: {numRowsRequired}')

        # Create a random starting point so that we don't always start at zero.
        trainingStartIndex = random.randint( 0, totalRowsInTrainingArea - numRowsRequired - 1 )
        #print(f'trainingStartIndex: {trainingStartIndex}')

        self.dfTrainFeatures = pd.read_csv(path, header=None, skiprows=trainingStartIndex, nrows=maxMoves)
        #print(f'dfTrainFeatures.shape: {self.dfTrainFeatures.shape}')
        assert self.dfTrainFeatures.shape[0] == maxMoves

        # Get Y data
        self.dfTrainY = pd.read_csv(pathY, header=None, skiprows=trainingStartIndex, nrows=maxMoves, names=['historic_close', 'future_close'])
        self.dfTrainY['gain'] = self.dfTrainY['future_close'] / self.dfTrainY['historic_close']

        if self.useNormalGain:
            self.dfTrainY['score'] = self.dfTrainY['gain']
        else:
            # Subracting 1 from 'gain' to make losses negative.
            self.dfTrainY['score'] = self.dfTrainY['gain'].sub(1)
        #print(f'dfTrainY.shape: {self.dfTrainY.shape}')
        assert self.dfTrainY.shape[0] == maxMoves





        #
        # Select testing data only from the last/second part of the data split.
        #
        totalRowsInTestingArea  = totalNumRows - totalRowsInTrainingArea

        # Create a random starting point so that we don't always start at the beginning.
        testingStartIndex = random.randint( 0, totalRowsInTestingArea - numRowsRequired - 1 )
        # Step past rows in training area.
        testingStartIndex += totalRowsInTrainingArea
        #print(f'testingStartIndex: {testingStartIndex}')

        self.dfTestFeatures = pd.read_csv(path, header=None, skiprows=testingStartIndex, nrows=maxMoves)
        #print(f'dfTestFeatures.shape: {self.dfTestFeatures.shape}')
        assert self.dfTestFeatures.shape[0] == maxMoves

        # Get Y data
        self.dfTestY = pd.read_csv(pathY, header=None, skiprows=testingStartIndex, nrows=maxMoves, names=['historic_close', 'future_close'])
        self.dfTestY['gain'] = self.dfTestY['future_close'] / self.dfTestY['historic_close']

        if self.useNormalGain:
            self.dfTestY['score'] = self.dfTestY['gain']
        else:
            # Subracting 1 from 'gain' to make losses negative.
            self.dfTestY['score'] = self.dfTestY['gain'].sub(1)
        #print(f'dfTestY.shape: {self.dfTestY.shape}')
        assert self.dfTestY.shape[0] == maxMoves
        #input('Press <Enter> to continue...')



    def getFeatures(self, train):
        """Get feature data."""
        if train:
            return self.dfTrainFeatures
        else:
            return self.dfTestFeatures

    def getPrices(self, train):
        """Get price data."""
        if train:
            return self.dfTrainY
        else:
            return self.dfTestY

    def getSize(self, train):
        """Get size of the data."""
        if train:
            return self.trainSize
        else:
            return self.test_features

