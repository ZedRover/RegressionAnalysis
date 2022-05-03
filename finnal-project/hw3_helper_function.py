import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def best_subset(y, X):
    from tqdm import tnrange, tqdm_notebook  # Importing tqdm for the progress bar
    RSS_list, R_squared_list, adj_R_squared_list, AIC_list, BIC_list, feature_list = [
    ], [], [], [], [], []
    numb_features = []
    k = len(X.columns)
    # Looping over k = 1 to k = 11 features in X
    # note that for python range(2) = 0,1
    for k in tnrange(1, len(X.columns) + 1, desc='Loop...'):

        # Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns, k):
            # we need to add constant term using sm.OLS
            X_c = sm.add_constant(X[list(combo)])
            # run the regression model
            model = sm.OLS(y, X_c).fit()
            # model.ssr is the sum of squared residuals
            RSS_list.append(model.ssr)
            R_squared_list.append(model.rsquared)
            adj_R_squared_list.append(model.rsquared_adj)
            AIC_list.append(model.aic)
            BIC_list.append(model.bic)
            feature_list.append(combo)
            numb_features.append(len(combo))

    # Store the results in DataFrame
    df_results = pd.DataFrame({'numb_features': numb_features, 'RSS': RSS_list, 'R_squared': R_squared_list,
                               'features': feature_list, 'adj_R_squared': adj_R_squared_list, 'AIC': AIC_list, 'BIC': BIC_list})
    return df_results


def forward_stepwise(y, X, remaining_features):
    features = []
    RSS_list, R_squared_list, adj_R_squared_list, AIC_list, BIC_list = [], [], [], [], []
    # Intialize feature list using dictionary. This is one way
    features_list = dict()
    k = len(remaining_features)
    for i in range(1, k+1):
        # initialize the best_RSS in each round to be infinity
        best_RSS = np.inf

        # iterate through all remaining features
        for combo in itertools.combinations(remaining_features, 1):

            # we need to add constant term using sm.OLS
            X_c = sm.add_constant(X[list(combo) + features])
            model = sm.OLS(y, X_c).fit()

            if model.ssr < best_RSS:                          # compare the RSS value with the smallest value in this round
                best_RSS = model.ssr                          # update the best value
                best_R_squared = model.rsquared               # update best best_R_squared
                # the best feature in this round
                best_feature = combo[0]
                best_aic = model.aic
                best_bic = model.bic
                best_adj_R_squared = model.rsquared_adj

        # Updating variables for next loop
        # add the best feature in the features set
        features.append(best_feature)
        # remove it from candidate set
        remaining_features.remove(best_feature)

        # Saving values for plotting
        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        AIC_list.append(best_aic)
        BIC_list.append(best_bic)
        adj_R_squared_list.append(best_adj_R_squared)
        features_list[i] = features.copy()

    # store results in df_results, which is a joint of df_features and df_values
    df_features = pd.DataFrame({'features': features_list})
    df_values = pd.DataFrame({'RSS': RSS_list, 'R_squared': R_squared_list,
                              'AIC': AIC_list, 'BIC': BIC_list, 'adj_R_squared': adj_R_squared_list})
    df_values.index += 1  # shift the index by 1 to get aligned with df_features
    df_results = pd.concat([df_features, df_values], axis=1, join='inner')
    df_results['numb_features'] = df_results.index
    return df_results


def plot_selection(df_results, standards):
    fig = plt.figure(figsize=(20, 6))

    for i, v in enumerate(standards):
        ax = fig.add_subplot(1, len(standards), i+1)
        ax.plot(df_results['numb_features'], df_results[v], color='lightblue')
        ax.scatter(df_results['numb_features'],
                   df_results[v], color='darkblue')
        if v == 'adj_R_squared':
            ax.plot(df_results[v].idxmax(), df_results[v].max(),
                    marker='x', markersize=20, color='r')
        else:
            ax.plot(df_results[v].idxmin(), df_results[v].min(),
                    marker='x', markersize=20, color='r')
        ax.set_xlabel('Number of predictors')
        ax.set_ylabel(v)

    fig.suptitle('Subset selection using ' + ", ".join(standards), fontsize=16)
    plt.show()
