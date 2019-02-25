import pandas as pd


class Dataloader(object): 

    def __init__(self, csv_path):

        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path)

        # Shuffle
        self.data.sample(frac=1.0, replace=True, random_state=1)

    def getHeader(self):
        return list(self.data.columns.values)

    def getData(self):
        # Split data into train, validation and test set with 80:10:10 ratio
        split_train = int(80 / 100 * len(self.data)) 
        split_val = int(90 / 100 * len(self.data)) 
        train = self.data[:split_train]
        val = self.data[split_train:split_val]
        test = self.data[split_val:]
        return train, val,  test

    def getFullData(self):
        return self.data
