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

def decision_tree(train, validate, test, target_var, depth=3, loop=False):
    x_train, y_train, x_validate, y_validate, X_test, Y_test = xy_train_validate_test(train,
                                                                                      validate,
                                                                                      test, 
                                                                                      target_var)
    models = {}
    y_pred = ''
    
    if loop:                                                                                 
        for num in range(1, depth + 1):
            clf = DecisionTreeClassifier(max_depth=num, random_state=123)
            clf = clf.fit(x_train, y_train)
            models[num] = clf
            
            # predictions on train observations
            y_pred = clf.predict(x_train)
            # probability for train observations
            y_pred_proba = clf.predict_proba(x_train)
            confu_matrix = confusion_matrix(y_train, y_pred)
            
            print(f'Classification Report for Tree with {num} depth on training set:\n {classification_report(y_train, y_pred)}')
            
            y_pred = clf.predict(x_validate)
            y_pred_proba = clf.predict_proba(x_validate)
            confu_matrix = confusion_matrix(y_validate, y_pred)

            print(f'Classification Report for Tree with {num} depth on validate set:\n {classification_report(y_validate, y_pred)}')
    else:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=123)
        clf = clf.fit(x_train, y_train)
        models[1] = clf
        y_pred = clf.predict(x_train)
        y_pred_proba = clf.predict_proba(x_train)
        confu_matrix = confusion_matrix(y_train, y_pred)
        print(f'Classification Report for Tree with {depth} depth on training set:\n {classification_report(y_train, y_pred)}')
        
        y_pred= clf.predict(x_validate)
        
        print(f'Classification Report for Tree with {depth} depth on validate set:\n {classification_report(y_validate, y_pred)}')

    