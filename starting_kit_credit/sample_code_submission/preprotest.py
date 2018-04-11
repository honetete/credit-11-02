from sys import path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager
model_dir = 'sample_code_submission/'
problem_dir = 'ingestion_program/'
score_dir = 'scoring_program/'
datadir = '../public_data'              # Change this to the directory where you put the input data
dataname = 'credit'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir);

from prepro import Preprocessor
input_dir = "../sample_data"
output_dir = "../resuts"


D = DataManager(dataname, datadir, replace_missing=True) # Load data
print("*** Original data ***")
print D

Prepro = Preprocessor()

# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])

# Here show something that proves that the preprocessing worked fine
print("*** Transformed data ***")
print D

# Preprocessing gives you opportunities of visualization:
# Scatter-plots of the 2 first principal components
# Scatter plots of pairs of features that are most relevant
import matplotlib.pyplot as plt
X = D.data['X_train']
Y = D.data['Y_train']
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
