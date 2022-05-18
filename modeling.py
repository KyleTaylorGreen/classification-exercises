import prepare
import acquire
import exploration
import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
import os
import split
import matplotlib.pyplot as plt
import itertools

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def impute_value(train, validate, test, col_names, strategy='most_frequent'):
    for col in col_names:
        imputer = SimpleImputer(missing_values= np.NaN, strategy=strategy)
        imputer = imputer.fit(train[[col]])

        train[[col]] = imputer.transform(train[[col]])
        validate[[col]] = imputer.transform(validate[[col]])
        test[[col]] = imputer.transform(test[[col]])

    return train, validate, test

def xy_train_validate_test(train, validate, test, target_var):
    
    x_train = train.drop(columns=[target_var])
    y_train = train[target_var]

    x_validate = validate.drop(columns=[target_var])
    y_validate = validate[target_var]

    X_test = test.drop(columns=[target_var])
    Y_test = test[target_var]

    return x_train, y_train, x_validate, y_validate, X_test, Y_test

def decision_tree(train, validate, test, target_var, depth=3):
    x_train, y_train, x_validate, y_validate, X_test, Y_test = xy_train_validate_test(train,
                                                                                      validate,
                                                                                      test, 
                                                                                      target_var)
    models = []
    y_pred = ''
    a = ''
    b = ''
                                                                             
    for num in range(1, depth + 1):
        clf = DecisionTreeClassifier(max_depth=num, random_state=123)
        clf = clf.fit(x_train, y_train)
        
        # predictions on train observations
        y_pred = clf.predict(x_train)
        a = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
        
        y_pred = clf.predict(x_validate)
        b = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
        
        models.append([a,b])

    return models

def make_model_results_into_df(models):
    a_df=  pd.DataFrame()
    #for i, model in enumerate(models):