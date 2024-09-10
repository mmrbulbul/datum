"""
Utility functions to check the distribution of train and test data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from sklearn.metrics import plot_roc_curve

except ImportError:
    from sklearn.metrics import RocCurveDisplay


def check_train_test(train_df, test_df, model=None, return_model=False):
    """
    Check whether train and test data comes from same distribution
    train_df: pandas dataframe without target column
    test_df: pandas dataframe without target column
    model: sklearn estimator, default is logistic regression
    return_model: bool, whether to return the fitted model
    """
    train_y = np.ones(len(train_df))
    test_y = np.zeros(len(test_df))

    combined_df = pd.concat([train_df, test_df])
    target = np.concatenate([train_y, test_y]).reshape(-1, 1)

    if model is None:
        model = LogisticRegression()
        
    model.fit(combined_df, target.ravel())

    pred = model.predict(combined_df)

    roc_scores = roc_auc_score(target, pred, average=None)
    try:
        plot_roc_curve(model, combined_df, target)
    except:
        RocCurveDisplay.from_estimator(model, combined_df, target)
    plt.show()
    print(roc_scores, np.mean(roc_scores))
    
    if return_model:
        return model
