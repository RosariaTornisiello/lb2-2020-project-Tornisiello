import numpy as np
import pandas as pd
import os 
import csv
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, make_scorer
import statistics 
import joblib


def padding(w, profile):

    """ pads the profile given in input: 
    adds a 0 matrix of lenght w/2 at the beginning 
    and at the end of the profile """

    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    pad = np.zeros((w//2, 20))
    pad = pd.DataFrame(pad, columns=aa)
    profile = pd.concat([pad, profile, pad])
    profile.reset_index(inplace=True, drop=True)
    return(profile)

def define_class(structure):

    """ converts secondary structure in class number
    H = 1, E = 2, C = 3 """

    if structure == 'H':
        structure= 1
    elif structure == 'E':
        structure= 2
    else: structure= 3
    return(structure)


def folds_dictionary(set_path):

    """ produces a dictionary with protein IDs ad keys 
    and the set number to which they belong as values """

    folds = dict()
    for filename in os.listdir(set_path):
        idList = [line.rstrip('\n') for line in open(set_path + filename, 'r')]
        for ID in idList:
            folds[ID] = filename[3]
    return(folds)

def generate_x_y(w, profile, ID, folds):

    """ generates feature matrix (x) and classes array
    (y) for a single profile """

    x = list()
    y = list()
    fold = list()
    i = folds[ID]
    start = 0
    end = (start + w)-1
    for _ in profile:
        window = profile[start:(end+1)]
        start += 1
        end += 1
        if len(window) == w:
            structure = window[w//2, 21]
            structure = define_class(structure)
            y.append(structure)
            window = np.delete(window, [20, 21], 1)
            window = window.flatten()
            x.append(window)
            fold.append(i)
    return(x, y, fold)

def split_X_y(w, profiles_path, folds):

    """ produces X_train, Y_train and test_fold (a np array 
    containing the set to which each sample belongs) """

    X_train = list()
    y_train = list()
    test_fold = list()
    for filename in os.listdir(profiles_path):
        ID = filename[0:-8]
        profile = pd.read_csv(profiles_path + filename, sep='\t')
        profile = padding(w, profile).to_numpy()
        profile_X, profile_y, fold = generate_x_y(w, profile, ID, folds)
        X_train += profile_X
        y_train += profile_y
        test_fold += fold
    X_train = np.array(X_train)
    y_train = pd.DataFrame(y_train).to_numpy().ravel()
    test_fold = np.array(test_fold)
    return(X_train, y_train, test_fold)

def myMCC(y_true, y_pred):

    """ computes one vs all MCC for each secondary structure
    and returnes the mean """
    
    ss = ['1', '2', '3']
    MCCs = []
    for structure in ss:
        y_true_temp = [structure if str(i) == structure else 'X' for i in y_true]
        y_pred_temp = [structure if str(i) == structure else 'X' for i in y_pred]
        MCC = matthews_corrcoef(y_true_temp, y_pred_temp)
        MCCs.append(MCC)
    return(statistics.mean(MCCs))

w = 17
set_path = '/home/rosaria/Desktop/LAB2/LAB2_project/cv/'
training_profiles_path = '/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/training_profile/'
#blind_profiles_path = '/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_profile/'

folds = folds_dictionary(set_path)
X_train, y_train, test_fold = split_X_y(w, training_profiles_path, folds)

my_scorer = make_scorer(myMCC)
ps = PredefinedSplit(test_fold)
mySVC = SVC(C=4, kernel='rbf', gamma=0.5, random_state=42)
cv_results = cross_validate(mySVC, X_train, y_train, scoring= my_scorer, cv=ps, return_estimator=True, verbose=10, n_jobs=5)

joblib.dump(cv_results, '/home/rosaria/Desktop/LAB2/LAB2_project/SVM/myCV_results')
print(cv_results)
