import unittest
from sklearn.ensemble import RandomForestRegressor
from random_forest import RandomForest

from prettytable import PrettyTable

import sys
import io

class TestModelMethods(unittest.TestCase):
    ''' Test the methods of the random forest model wrapper.
    
    Arguments:
        unittest -- Python Unit Test Framework
    '''


    def setUp(self):
        ''' Test runner setup.
        '''

        self.rf = RandomForest()     
        self.rf.model_file = 'testModel.pth'

    def testDataVariables(self):
        ''' Test whether feature and target variables are defined in the model.
        '''

        self.assertGreater(len(self.rf.features), 0)
        self.assertGreater(len(self.rf.target), 0)

    def testDataLoading(self):
        ''' Test the data loading.
        '''

        for dataset in ['train', 'val', 'test', 'full']:
            self.assertGreater(len(self.rf.data[dataset]), 0)
            self.assertGreater(len(self.rf.samples[dataset]), 0)
            self.assertGreater(len(self.rf.labels[dataset]), 0)

    def testSaveExistingModel(self):
        ''' Test the save method for a blank model.
        '''

        self.rf.model = RandomForestRegressor()
        self.assertTrue(self.rf._saveModel())
    
    def testSaveNonExistingModel(self):
        ''' Test the save method for a non existing model.
        '''

        self.rf.model = None
        self.assertFalse(self.rf._saveModel())

    def testLoadExisitngModel(self):
        ''' Test the load model for a previous saved model.
        '''

        self.rf._saveModel()
        self.assertTrue(self.rf._loadModel())

    def testLoadNonExisitngModel(self):
        ''' Test the load method for a non-existing model path.
        '''

        self.rf.model_file = 'non-existing/path.x'
        self.assertFalse(self.rf._loadModel())

if __name__ == '__main__':
    unittest.main()