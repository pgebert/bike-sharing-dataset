import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import random
# Sklearn models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from prettytable import PrettyTable
import matplotlib.pyplot as plt

import pickle
import os.path

from dataloader import Dataloader

class RandomForest():
    ''' Random forest data model wrapper to provide addtional methods for the Bike Sharing Dataset.
    
    '''


    def __init__(self, data_path='Bike-Sharing-Dataset/hour.csv'):
        '''Initialize the random forest model.
        
        Keyword Arguments:
            data_path {str} -- Path to the Bike Sharing Dataset. (default: {'Bike-Sharing-Dataset/hour.csv'})            
        '''

        # Make results reproducible
        random.seed(100)
        
        # Load data form bike sharing csv
        self.data = {}
        dataloader = Dataloader(data_path)
        self.data['full'] = dataloader.getFullData()

        # Define feature and target variables
        self.features= ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        self.target = ['cnt']

        # Convert pandas frame into samples and labels
        self.samples, self.labels = {}, {}
        self.samples['full'] = self.data['full'][self.features].values
        self.labels['full'] = self.data['full'][self.target].values.ravel()    

        # Define model 
        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=4,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
           oob_score=False, random_state=100, verbose=0, warm_start=False)

    def _saveModel(self, model_file='model.pth'):
        ''' Store the random forest model on the disk.

        Keyword Arguments:
            model_file {str} -- Model path (default: {'model.pth'})        
        Returns:
            boolean -- success of data storage
        '''

        success = False
        if self.model is not None:
            pickle.dump(self.model, open(model_file, 'wb'))
            success = True
        return success


    def _loadModel(self, model_file='model.pth'):
        ''' Loads the random forest model from the disk.

        Keyword Arguments:
            model_file {str} -- Model path (default: {'model.pth'})        
        Returns:
            boolean -- success of data loading
        '''

        success = False
        if os.path.exists(model_file):
            self.model = pickle.load(open(model_file, 'rb'))
            success = True
        return success

    def randomizedParameterSearch(self, iter=100):
        ''' Defines a parameter grid and performs a random search using three fold cross validation to estimate 
        the best parameter set for the random forest data model.
        
        Keyword Arguments:
            iter {int} -- Number of search iterations. (default: {100})        
        Returns:
            [dict of str] -- Dictionary with the best random forest parameters found in this search.
        '''


        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        self.model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =iter, cv = 3, verbose=2, random_state=0, n_jobs=-1)

        return self.model.get_params()

    def train(self, samples, labels):
        '''Train the random forest model on the training data.
        
        Keyword Arguments:
            samples {pandas data frame} -- Training samples
            labels {list of int} -- Training labels
        '''

        assert(len(samples) == len(labels))
        self.model.fit(samples, labels)        

    def test(self, samples, labels):
        ''' Evaluate the random forest performance on the test data.
        
        Keyword Arguments:
            samples {pandas data frame} -- Test samples
            labels {list of int} -- Test labels
        Returns:
            [dict of int] -- Dictionary with the test results.
        '''

         # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return

        assert(len(samples) == len(labels)) 

        pred = self.model.predict(samples)

        mse = mean_squared_error(labels, pred)
        mae = mean_absolute_error(labels, pred)
        score = self.model.score(samples, labels)    
        rmsle = np.sqrt(mean_squared_log_error(labels, pred))

        return { 'mse': mse, 'mae': mae, 'score': score, 'rmsle': rmsle}

    def kFoldCrossvalidation(self): 
        ''' Runs a three-fold cross-validation on the Bike Sharing dataset.
        '''


        table = PrettyTable()
        table.field_names = ["Model", "Split", "Mean Squared Error", "Mean Absolute Error", 'RMSLE', "R² score"]

        kf = KFold(n_splits=3, shuffle=True, random_state=100)

        res = []
        split = 1
        
        for train_index, test_index in kf.split(self.samples['full']):

            samples, labels = {}, {}
            samples['train'], labels['train'] = self._selectData(train_index)
            samples['test'], labels['test'] = self._selectData(test_index)            

            self.train(samples['train'], labels['train'])
            res.append(self.test(samples['test'], labels['test']))    
            table.add_row([type(self.model).__name__, split, format(res[-1]['mse'], '.2f'), format(res[-1]['mae'], '.2f'), format(res[-1]['rmsle'], '.2f'), format(res[-1]['score'], '.2f')])
            split += 1

        mse = np.mean([item['mse'] for item in res])
        mae = np.mean([item['mae'] for item in res])
        score = np.mean([item['score'] for item in res])
        rmsle = np.mean([item['rmsle'] for item in res])

        table.add_row([type(self.model).__name__, 'Mean', format(mse, '.2f'), format(mae, '.2f'), format(rmsle, '.2f'), format(score, '.2f')])
        print(table)

      def _selectData(self, index_list): 
        ''' Filters the full dataset depending on the indices in the provided index list.
        
        Arguments:
            index_list {list of int} --  List with sample indices to keep.
        Returns:
            samples -- samples with the provided indices
            labels -- labels with the provided indices
        '''

        samples = [self.samples['full'][i] for i in index_list]
        labels = [self.labels['full'][i] for i in index_list]

        return samples, labels


    def featureImportances(self):
        ''' Print the feature importances of a trained random forest model.        
        '''

        # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return

        # Set first split as default training data
        kf = KFold(n_splits=3, shuffle=True, random_state=100)        
        train_index, test_index = next(kf.split(self.samples['full']))
        samples, labels = self._selectData(train_index) 

        self.model.fit(samples, labels)

        # Get sorted feature importances from model
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(len(self.features)):
            print("%d. feature %s (%f)" % (f + 1, self.features[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.features)), importances[indices], color="cornflowerblue", yerr=std[indices], align="center")
        plt.xticks(range(len(self.features)), [self.features[i] for i in indices])
        plt.show()

    
if __name__ == "__main__":

    model = RandomForest()
    # model.train()
    # model._saveModel()
    # model._loadModel()
    model.kFoldCrossvalidation()
    # model.featureImportances()

