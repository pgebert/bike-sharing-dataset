import unittest
from sklearn.ensemble import RandomForestRegressor
from random_forest import RandomForest

from prettytable import PrettyTable

import sys
import io

class TestModelMethods(unittest.TestCase):

    def setUp(self):
        self.rf = RandomForest()     
        self.rf.model_file = 'testModel.pth'

    def tearDown(self):
        pass        

    def testDataVariables(self):
        self.assertGreater(len(self.rf.features), 0)
        self.assertGreater(len(self.rf.target), 0)

    def testDataLoading(self):
        for dataset in ['train', 'val', 'test', 'full']:
            self.assertGreater(len(self.rf.data[dataset]), 0)
            self.assertGreater(len(self.rf.samples[dataset]), 0)
            self.assertGreater(len(self.rf.labels[dataset]), 0)

    def testSaveExistingModel(self):
        self.rf.model = RandomForestRegressor()
        self.assertTrue(self.rf._saveModel())
    
    def testSaveNonExistingModel(self):
        self.rf.model = None
        self.assertFalse(self.rf._saveModel())

    def testLoadExisitngModel(self):
        self.rf._saveModel()
        self.assertTrue(self.rf._loadModel())

    def testLoadNonExisitngModel(self):
        self.rf.model_file = 'non-existing/path.x'
        self.assertFalse(self.rf._loadModel())

if __name__ == '__main__':
    unittest.main()