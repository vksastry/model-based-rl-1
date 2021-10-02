import numpy as np
import pandas as pd

def convert():
    # Convert text files to numpy files.
    path = '../data/XTotal1.at'
    pathY = path.replace(".at", ".price.at")

    dfFeatures = pd.read_csv(path, header=None)
    npFeatures = dfFeatures.to_numpy()
    pathOut = path.replace(".at", ".at.gz")
    pathOut = path.replace(".at", ".at.txt")
    np.savetxt(pathOut, npFeatures, delimiter=',', newline='\n')


    """
    dfY = pd.read_csv(pathY, header=None, names=['historic_close', 'future_close'])
    dfY['gain'] = dfY['future_close'] / dfY['historic_close']

    useNormalGain = True
    if useNormalGain:
        dfY['score'] = dfY['gain']
    else:
        # Subracting 1 from 'gain' to make losses negative.
        dfY['score'] = dfY['gain'].sub(1)


    # Save all of dfFeatures

    # Save dfY but only one column.  But what to do with score?
    # Can this be turned into a binary classification problem?
    # Does it need to be?  The previous version was only based on gain.
    # Saving all of dfY worked in the past and should work now.

    npY = dfY.to_numpy()

    pathOutY = pathOut.replace(".at", ".price.at")

    #numpy.savetxt(pathOut, npFeatures, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
    np.savetxt(pathOutY, npY, delimiter=',', newline='\n')
    """



if __name__ == '__main__':
    convert()