#import the model trained by ste with gpu on the entire training set with C=2 and gamma=0.5
#prepare X_test
#predict(X_test)

import numpy as np
import pandas as pd
import os 
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

def generate_x(w, profile):

    """ generates feature matrix (x) for a single profile """

    x = list()
    start = 0
    end = (start + w)-1
    for _ in profile:
        window = profile[start:(end+1)]
        start += 1
        end += 1
        if len(window) == w:
            window = np.delete(window, [20, 21], 1)
            window = window.flatten()
            x.append(window)
    return(x)

def compute_X_test(w, profiles_path):

    """ produces X_test """

    X_test = list()
    for filename in os.listdir(profiles_path):
        profile = pd.read_csv(profiles_path + filename, sep='\t')
        profile = padding(w, profile).to_numpy()
        profile_X = generate_x(w, profile)
        X_test += profile_X
    X_test = np.array(X_test)
    return(X_test)

w = 17
blind_profile_path = '/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/blind_test_dataset/blind_profile/'
X_test = compute_X_test(w, blind_profile_path)
X_test = X_test.astype(np.float)

best_model = joblib.load('/home/rosaria/Desktop/LAB2/refitted_WHOLE_C2_gamma02.joblib')
best_model.set_params(max_mem_size=100, n_jobs=1, cache_size=100)
y_pred = best_model.predict(X_test)
# with open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/SVM_results/y_pred_blind.txt', 'w') as f:
#     print(y_pred.shape, file=f)