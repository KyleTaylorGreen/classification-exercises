import pandas as pd
import numpy as np
import seaborn as sns
import os
import acquire
import prepare
import split
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats

alpha = 0.05


def quantitative_hist_boxplot_describe(training_df, quantitative_col_names,separate=True):
    for col in quantitative_col_names:
        training_df[col].hist()
        plt.xlabel(col)
        plt.show()
    
    if separate:
        for col in quantitative_col_names:
            training_df.boxplot(column=col)
            plt.show()
    else:    
        training_df.boxplot(column=quantitative_col_names)
        plt.show()

    print(training_df[quantitative_col_names].describe().T)
    


def target_freq_hist_count(training_df, target_col):
    training_df[target_col].hist()
    print(training_df[target_col].value_counts())
    plt.show()


def odd(num):
    if num % 2 != 0:
        return True
    else:
        return False
    
def even(num):
    return not odd(num)

def find_subplot_dim(quant_col_lst):
    
    # goal: make x 
    # checks if len is even (making 2 rows)
    if even(len(quant_col_lst)):
        length = len(quant_col_lst)
    else:
        length = len(quant_col_lst) + 1
        
    divided_by_2 = int(length/ 2)
    divided_by_other_factor = int(length / divided_by_2)
    subplot_dim = [divided_by_2, divided_by_other_factor]
    
    return subplot_dim

def quant_vs_target_bar(train_df, target_col, quant_col_lst, mean_line=False):
    
    subplot_dim = find_subplot_dim(quant_col_lst)
    
    plots = []
    fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], sharex=True, figsize=(10,5))
    
    for axe in axes:
        for ax in axe:
            plots.append(ax)

    for n in range(len(quant_col_lst)):    
        sns.barplot(ax=plots[n], x=train_df[target_col], y =train_df[quant_col_lst[n]])
        
        if mean_line:
            avg = train_df[quant_col_lst[n]].mean()
            plots[n].axhline(avg,  label=f'Avg {train_df[quant_col_lst[n]]}')

def describe_quant_grouped_by_target(training_df, quantitative_col, 
                                     target_col):
    lst_cpy = quantitative_col[:]
    lst_cpy.append(target_col)
    
    print(training_df[lst_cpy].groupby(target_col).describe().T)


def target_subsets(target_col, training_df):
    
    values = training_df[target_col].unique()
    subset_dict= {}
    
    for val in values:
        subset_dict[val] = training_df[training_df[target_col]==val]
        
    return subset_dict

def combinations_of_subsets(target_col, training_df):
    subsets = target_subsets(target_col, training_df)
    combos = list(itertools.combinations(subsets.keys(), 2))
    
    return subsets, combos

def mannshitneyu_for_quant_by_target(target_col, training_df, 
                                    quantitative_col):
    
    predictors = {}
    subsets, combos = combinations_of_subsets(target_col, training_df)
    p_exceeds_alpha = []
        

    for i, pair in enumerate(combos):
        
        #print(f'{pair[0]}/{pair[1]}:' )
        predictors[str(pair)] = []
        for col in quantitative_col:
            t, p = stats.mannwhitneyu(subsets[pair[0]][col], 
                                      subsets[pair[1]][col])
            #print(f'{pair[0]}/{pair[1]} {col}:')
            #print(f't: {t}, p: {p}\n')
            
            if p < alpha:
                predictors[str(pair)].append({col: [t, p]})
            else:
                p_exceeds_alpha.append([str(pair), col, t, p])
                
                
    return subsets, predictors, p_exceeds_alpha, combos
            
    
def print_mannswhitneyu_predictors(predictors):
    for keys, values in predictors.items():
        print(keys)
        for value in values:
            print(value)
        print()
    
def print_mannswhitneyu_failures(p_exceeds_alpha):
    for val in p_exceeds_alpha:
        print(f'Combination: {val[0]}')
        print(f'Measurement: {val[1]}')
        print(f't: {val[2]}, p: {val[3]}')
        print()
        

def print_quant_by_target(target_col, training_df, quant_col):
    subsets, predictors, p_exceeds_alpha, combos = mannshitneyu_for_quant_by_target(target_col, 
                                                                            training_df, 
                                                                            quant_col)
    print_mannswhitneyu_predictors(predictors)
    print_mannswhitneyu_failures(p_exceeds_alpha)
    
    combo_predic = {}
    for combo in combos:
        combo_predic[combo] = []
        #print(predictors[str(combo)])
        for predic in predictors[str(combo)]:
              #print(list(predic.keys())[0])
              combo_predic[combo].append(list(predic.keys())[0])

    return subsets, predictors, p_exceeds_alpha, combo_predic

def two_quants_by_target_var(target_col, training_df, combo_predic, 
                         subtitle=""):
    
    for combo in combo_predic.keys():
        subplot_dim= find_subplot_dim(combo_predic[combo])
        
    
        plots = []
        fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], sharex=True, figsize=(10,5))
        
        for axe in axes:
            for ax in axe:
                plots.append(ax)
                
        predictors_comb = list(itertools.combinations(combo_predic[combo], 2))
    
        for i, pair in enumerate(predictors_comb):
            sns.scatterplot(x=training_df[pair[0]], y=training_df[pair[1]],
                           hue=training_df[target_col],
                           ax= plots[i])
        plt.show()

def overview(train_df,
             quant_cols,
             target_var):
    quantitative_hist_boxplot_describe(train_df, quant_cols,separate=True)
    target_freq_hist_count(train_df, target_var)
    quant_vs_target_bar(train_df, target_var, quant_cols, mean_line=True)
    describe_quant_grouped_by_target(train_df, quant_cols, target_var)

    subsets, predictors, p_exceeds_alpha, combos = print_quant_by_target(target_var, train_df, quant_cols)
    two_quants_by_target_var(target_var, train_df, combos)

