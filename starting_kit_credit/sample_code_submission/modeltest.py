from sys import path
path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need

from data_manager import DataManager
from data_converter import convert_to_num 
from model import model
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

metric_name = 'auc_metric_'

input_dir = "../public_data" # A remplacer par public_data
output_dir = "../res"

basename = 'credit'
D = DataManager(basename, input_dir) # Load data
#print D
print "Data loaded"
M = model()
M.__init__()

#model_dir = 'sample_code_submission/'
#trained_model_name = model_dir + basename    
#M = M.load(trained_model_name)

# Train
Yonehot_tr = D.data['Y_train']
# Attention pour les utilisateurs de problemes multiclasse,
# mettre convert_to_num DANS la methode fit car l'ingestion program
# fournit Yonehot_tr a la methode "fit"
# Ceux qui resolvent des problemes a 2 classes ou des problemes de
# regression n'en ont pas besoin
Ytrue_tr = convert_to_num(Yonehot_tr, verbose=False) # For multi-class only, to be compatible with scikit-learn
M.fit(D.data['X_train'], Ytrue_tr)

# Making predictions
Ypred_tr = M.predict(D.data['X_train'])
Ypred_va = M.predict(D.data['X_valid'])
Ypred_te = M.predict(D.data['X_test'])  

X_train = D.data['X_train']
Y_train = D.data['Y_train']
# Cross-validation predictions
print("Cross-validating")
from sklearn.model_selection import KFold
from numpy import zeros  
n = 10   # 10-fold cross-validation
kf = KFold(n_splits=n)
kf.get_n_splits(X_train)
Ypred_cv = zeros(Ypred_tr.shape)
i=1
for train_index, test_index in kf.split(X_train):
    print("Fold{:d}".format(i))
    Xtr, Xva = X_train[train_index], X_train[test_index]
    Ytr, Yva = Y_train[train_index], Y_train[test_index]
    M.fit(Xtr, Ytr)
    Ypred_cv[test_index] = M.predict(Xva)
   
    i = i+1
    
# Compute and print performance
training_score = metrics.roc_auc_score(Y_train, Ypred_tr)
cv_score = metrics.roc_auc_score(Y_train, Ypred_cv)
fpr, tpr, c = metrics.roc_curve(Y_train,Ypred_cv)
# a =fpr
#b = tpr

line = np.linspace(0,1)

plt.plot(fpr,tpr,'g', label='roc curve using cross-valid')
plt.plot(line,line,'b-')

plt.ylabel("True Positive rate")
plt.xlabel("False Possitive rate")
plt.legend()
plt.show()
    

print("\nRESULTS FOR SCORE {:s}".format(metric_name))
print("TRAINING SCORE= {:f}".format(training_score))
print("CV SCORE= {:f}".format(cv_score))



"""
# We can compute the training success rate 
acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
# But it might be optimistic compared to the validation and test accuracy
# that we cannot compute (except by making submissions to Codalab)
# So, we can use cross-validation:
acc_cv = cross_val_score(M, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')

print "One sigma error bars:"
print "Training Accuracy = %5.2f +-%5.2f" % (acc_tr, ebar(acc_tr, Ytrue_tr.shape[0]))
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (acc_cv.mean(), acc_cv.std())
"""