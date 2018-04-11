
from sys import argv
import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from prepro import Preprocessor
from sklearn.pipeline import Pipeline
import xgboost as xgb                                                                                                                                                                                                                                         

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

# variable contenant le classifier
class model(BaseEstimator):
    def __init__ (self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.prepro = Preprocessor()
        self.clf = xgb.XGBRegressor()
        

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
        
    
    def predict(self, X,seuil):
       '''
       This fonction should predict the value of each X
       args:
           X: Training data matrix of dim num_train_samples * num_feat
           sueil: biais w
        
       '''
       Preprocessor.transform(self.prepro,X)
       pred = self.clf.predict(X)
       #print pred
       y_pred = np.zeros(len(pred))

#       for i in range(len(pred)):
#           if pred[i]<=seuil:
#               y_pred[i]=0
 #           else:
 #              y_pred[i]=1
           
       print("Prediction Done")
       #print (y_pred.shape)
       return pred#y_pred


    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        self = pickle.load(open(modelfile))
        print("Model reloaded from: " + modelfile)
        return self
        
    
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    # We can use this to run this file as a script and test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data" # A remplacer par public_data
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
                         
    from sklearn.metrics import accuracy_score      
    # Interesting point: the M2 prepared challenges using sometimes AutoML challenge metrics
    # not scikit-learn metrics. For example:
#    from libscores import bac_metric
#    from libscores import auc_metric
                 
    from data_manager import DataManager 
    from data_converter import convert_to_num 
    
    basename = 'credit'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    # Here we define 3 classifiers and compare them
#    classifier_dict = {
#            'Pipeline': Pipeline([('prepro', Preprocessor()), ('classif', clasif)]),
#        
#    }
#    for key in classifier_dict:
    myclassifier = model()
 
    # Train
    Yonehot_tr = D.data['Y_train']
    # Attention pour les utilisateurs de problemes multiclasse,
    # mettre convert_to_num DANS la methode fit car l'ingestion program
    # fournit Yonehot_tr a la methode "fit"
    # Ceux qui resolvent des problemes a 2 classes ou des problemes de
    # regression n'en ont pas besoin
    Ytrue_tr = convert_to_num(Yonehot_tr, verbose=False) # For multi-class only, to be compatible with scikit-learn
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    myclassifier.fit(D.data['X_train'], Ytrue_tr)
    
    # Some classifiers and cost function use a different encoding of the target
    # values called on-hot encoding, i.e. a matrix (nsample, nclass) with one at
    # the position of the class in each line (also called position code):
    #nclass = len(set(Ytrue_tr))
    #Yonehot_tr = np.zeros([Ytrue_tr.shape[0],nclass])
    #for i, item in enumerate(Ytrue_tr): Yonehot_tr[i,item]=1

    # Making classification predictions (the output is a vector of class IDs)
    Ypred_tr = myclassifier.predict(D.data['X_train'],0.5)
    Ypred_va = myclassifier.predict(D.data['X_valid'],0.5)
    Ypred_te = myclassifier.predict(D.data['X_test'],0.5)  
    
    print("Cross-validating")
    from sklearn.model_selection import KFold
    from numpy import zeros  
    n = 10 # 10-fold cross-validation
    kf = KFold(n_splits=n)
    kf.get_n_splits(X_train)
    Ypred_cv = zeros(Ypred_tr.shape)
    i=1
    for train_index, test_index in kf.split(X_train):
        print("Fold{:d}".format(i))
        Xtr, Xva = X_train[train_index], X_train[test_index]
        Ytr, Yva = Y_train[train_index], Y_train[test_index]
        myclassifier.fit(Xtr, Ytr)
        Ypred_cv[test_index] = myclassifier.predict(Xva,0.5)
        i = i+1
                

    # Training success rate and error bar:
    # First the regular accuracy (fraction of correct classifications)
    from sklearn.metrics import roc_auc_score
    print roc_auc_score(Ytrue_tr, Ypred_cv)
    plt.plot(Ytrue_tr, Ypred_cv)
    
    # Note that the AutoML metrics are rescaled between 0 and 1.
    
#    print "%s\t%5.2f\t(%5.2f)" % (key, acc, ebar(acc, Ytrue_tr.shape[0]))
#    print "The error bar is valid for Acc only"
    # Note: we do not know Ytrue_va and Ytrue_te
    # See modelTest for a better evaluation using cross-validation 
        
    # Another useful tool is the confusion matrix
    from sklearn.metrics import confusion_matrix
#    print "Confusion matrix for %s" % key
    print confusion_matrix(Ytrue_tr, Ypred_tr)
