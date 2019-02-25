import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import random
# Sklearn models
from sklearn.ensemble import RandomForestRegressor

from prettytable import PrettyTable
import matplotlib.pyplot as plt

import pickle
import os.path

from dataloader import Dataloader

class RandomForest():
    ''' Random forest data model wrapper to provide addtional methods for the Bike Sharing Dataset.
    
    '''


    def __init__(self, data_path='Bike-Sharing-Dataset/hour.csv', model_file='model.pth'):
        '''Initialize the random forest model.
        
        Keyword Arguments:
            data_path {str} -- Path to the Bike Sharing Dataset. (default: {'Bike-Sharing-Dataset/hour.csv'})
            model_file {str} -- In-/Out-path for the random forest model. (default: {'model.pth'})
        '''

        # Make results reproducible
        random.seed(100)
        
        # Load data form bike sharing csv
        self.data = {}
        dataloader = Dataloader(data_path)
        self.data['train'], self.data['val'], self.data['test'] = dataloader.getData()
        self.data['full'] = dataloader.getFullData()

        # Define feature and target variables
        self.features= ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        self.target = ['cnt']

        # Get samples and labels
        self.samples, self.labels = {}, {}
        for dataset in ['train', 'val', 'test', 'full']:
            self.samples[dataset] = self.data[dataset][self.features].values
            self.labels[dataset] = self.data[dataset][self.target].values.ravel()        

        # Define model 
        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=4,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

        # Model path
        self.model_file = model_file

    def _saveModel(self):
        ''' Store the random forest model on the disk.
        
        Returns:
            boolean -- success of data storage
        '''

        success = False
        if self.model is not None:
            pickle.dump(self.model, open(self.model_file, 'wb'))
            success = True
        return success


    def _loadModel(self):
        ''' Load random forest model from the disk.
        
        Returns:
            boolean -- success of model loading
        '''


        success = False
        if os.path.exists(self.model_file):
            self.model = pickle.load(open(self.model_file, 'rb'))
            success = True
        return success

    def _evaluateOnDataset(self, dataset, table):
        ''' Evaluate data model on a given dataset [train, val, test, full].
        
        Arguments:
            dataset {str} -- Dataset for evaluation. Possibilities: train, val, test, full
            table {[type]} -- Pretty table with four columns to store the results
        
        '''


         # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return

        pred = self.model.predict(self.samples[dataset])

        mse = mean_squared_error(self.labels[dataset], pred)
        score = self.model.score(self.samples[dataset], self.labels[dataset])    
        rmsle = np.sqrt(mean_squared_log_error(self.labels[dataset], pred))

        table.add_row([type(self.model).__name__, dataset, format(mse, '.2f'), format(rmsle, '.2f'), format(score, '.2f')])

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

        # Train model with best parameters
        self.model.fit(self.samples['train'], self.labels['train'])

        return self.model.get_params()

    def train(self):
        ''' Train the random forest model on the training data.
        '''
        # Train model
        self.model.fit(self.samples['train'], self.labels['train'])

    def evaluate(self):
        ''' Evaluate the random forest performance on the train and validation set.
        '''

        table = PrettyTable()
        table.field_names = ["Model", "Dataset", "Mean Squared Error", 'RMSLE', "R² score"]

        self._evaluateOnDataset('train', table)
        self._evaluateOnDataset('val', table)

        print(table)

    def featureImportances(self):
        ''' Print the feature importances of a trained random forest model.        
        '''


        # Check model loaded
        if self.model is None:
            print("Please load or train a model before!")
            return

        # Get sorted feature importances from model
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(self.samples['val'].shape[1]):
            print("%d. feature %s (%f)" % (f + 1, self.features[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.samples['val'].shape[1]), importances[indices], color="cornflowerblue", yerr=std[indices], align="center")
        plt.xticks(range(self.samples['val'].shape[1]), [self.features[i] for i in indices])
        plt.xlim([-1, self.samples['val'].shape[1]])
        plt.show()

    
if __name__ == "__main__":

    model = RandomForest()
    # model.train()
    # model._saveModel()
    model._loadModel()
    model.evaluate()
    # model.featureImportances()

