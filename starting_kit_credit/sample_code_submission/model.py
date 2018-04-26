

import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from prepro import Preprocessor

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

class model(BaseEstimator):
    def __init__ (self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.prepro = Preprocessor()
        self.clf = GradientBoostingRegressor(n_estimators=250)
        

    def fit(self,X,Y) :
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.

        '''
        
        Preprocessor.fit_transform(self.prepro,X)
        self.clf = self.clf.fit(X,Y)
        print("Fit Done")
        
    
    def predict(self, X):
       '''
       This fonction should predict the value of each X
       args:
           X: Training data matrix of dim num_train_samples * num_feat
           sueil: biais w
        
       '''
       Preprocessor.transform(self.prepro,X)
       pred = self.clf.predict(X)
       """
       y_pred = np.zeros(len(pred))
       for i in range(len(y_pred)):
           y_pred[i] = pred[i,1]
           
       print(y_pred)
       """
       print ("Predict Done")
       return pred#y_pred


    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        self = pickle.load(open(modelfile))
        print("Model reloaded from: " + modelfile)
        return self
        
