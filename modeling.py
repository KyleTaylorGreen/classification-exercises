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

from sklearn.ensemble import RandomForestClassifier
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

def random_forests(train, validate, test, target_var, 
                   min_sample_leaf=3, depth=3, inverse=False):
    x_train, y_train, x_validate, y_validate, X_test, Y_test = xy_train_validate_test(train,
                                                                                      validate,
                                                                                      test, 
                                                                                      target_var)
    models = []
    reports = []
                                                                             
    for num in range(1, depth + 1):
        # Have min_sample_leafs decrease as depth increases
        if inverse:
            min_sample_leaf = abs((depth+1)-num)

        clf = RandomForestClassifier(max_depth=num, min_samples_leaf=min_sample_leaf, random_state=123)
        models.append(fit_score_train_validate(clf, num, x_train, x_validate, 
                                               y_train, y_validate,
                                               min_sample_leaf)[0])
        reports.append(fit_score_train_validate(clf, num, x_train, x_validate, y_train, y_validate)[1])
        
    models = make_model_results_into_df(models)
    
    return models, reports

def decision_tree(train, validate, test, target_var, depth=3):
    x_train, y_train, x_validate, y_validate, X_test, Y_test = xy_train_validate_test(train,
                                                                                      validate,
                                                                                      test, 
                                                                                      target_var)
    models = []
    reports = []
                                                                             
    for num in range(1, depth + 1):
        clf = DecisionTreeClassifier(max_depth=num, random_state=123)
        models.append(fit_score_train_validate(clf, num, x_train, x_validate, y_train, y_validate)[0])
        reports.append(fit_score_train_validate(clf, num, x_train, x_validate, y_train, y_validate)[1])
    models = make_model_results_into_df(models)

    return models, reports


def fit_score_train_validate(clf, num, x_train, x_validate, 
                             y_train, y_validate,
                             min_sample_leaf=""):
    clf = clf.fit(x_train, y_train)
    results = []
    
    # predictions on train observations
    y_pred_train = clf.predict(x_train)
    train_score = clf.score(x_train, y_train)
    #print(train_score)
    train_class_report = classification_report(y_train, y_pred_train, output_dict=True)
    train_class_report = pd.DataFrame(train_class_report).T

    y_pred_validate = clf.predict(x_validate)
    validate_score = clf.score(x_validate, y_validate)
    #print(validate_score)
    
    validate_class_report = classification_report(y_validate, y_pred_validate, output_dict=True)
    validate_class_report = pd.DataFrame(validate_class_report).T

    results = make_model_stats(num, train_score, validate_score, 
                               train_class_report, validate_class_report,
                               min_sample_leaf)

    reports = {'train_report': train_class_report, 'validate_report': validate_class_report}

    return results, reports

def make_model_stats(num, train_score, validate_score, 
                     train_class_report, validate_class_report,
                     min_sample_leaf):

    results = ({'depth': num,
                'min_samples_leaf': min_sample_leaf, 
                'train_accuracy': train_score, 
                'validate_accuracy': validate_score, 
                'difference': train_score - validate_score,
                'percent_diff': round((train_score - validate_score) / train_score * 100, 2),
                'classification_report_validate': validate_class_report,
                'classification_report_train': train_class_report
              })
    

    return results


def make_model_results_into_df(models):
    models= pd.DataFrame(models)
    
    return models

def summary_results(models):
    summary_cols = ['depth', 'train_accuracy', 'validate_accuracy', 
                    'difference', 'percent_diff']
    if 'min_samples_leaf' in models.columns:
        summary_cols.insert(1, 'min_samples_leaf')
    return models[summary_cols]

def difference_means(models):
    differences = {'difference_mean': models.difference.mean(),
                   'percent_diff_mean': models.percent_diff.mean()}

    return differences

    
class Results:

    def __init__(self, models_df, reports):
        self.reports = reports
        self.df =models_df
        self.columns = self.df.columns
        self.diff_means = difference_means(self.df)
        self.summary = summary_results(self.df)

    def report(self, index):
        for i, report in enumerate(self.reports):
            if i == index:
                print(f'Training report for index {i}:\n', report["train_report"], '\n')
                print(f'Validate report for index {i}:\n', report["validate_report"], '\n')
    
    def by_min_sample_leaf_equals(self, N, inclusive=False):
        return self.df[self.df.min_samples_leaf == N]

    def by_min_sample_leaf_gtr_than(self, N, inclusive=False):
        if inclusive:
            return self.df[self.df.min_samples_leaf >= N]
        else:
            return self.df[self.df.min_samples_leaf > N]

    def by_min_sample_leaf_less_than(self, N, inclusive=False):
        if inclusive:
            return self.df[self.df.min_samples_leaf <= N]
        else:
            return self.df[self.df.min_samples_leaf < N]

    def by_depth_equals(self, depth):
        return self.df[self.df.depth == depth]

    def by_depth_gtr_than(self, depth, inclusive=False):
        
        if inclusive:
            return self.df[self.df.depth >= depth]
        else:
            return self.df[self.df.depth >  depth]
    
    def by_depth_less_than(self, depth, inclusive=False):
        if inclusive:
            return self.df[self.df.depth <= depth]
        else:
            return self.df[self.df.depth < depth]

    def by_percent_diff_less_than(self, num, inclusive=False):
        if inclusive:
            return self.df[self.df.percent_diff <= num]
        else:
            return self.df[self.df.percent_diff < num]
    
    def by_percent_diff_gtr_than(self, num, inclusive=False):
        if inclusive:
            return self.df[self.df.percent_diff >= num]
        return self.df[self.df.percent_diff > num]

    

    

