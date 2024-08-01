"""
Utility functions to check the distribution of train and test data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, roc_auc_score


def check_train_test(train_df, test_df):
    train_y = np.ones(len(train_df))
    test_y = np.zeros(len(test_df))

    combined_df = pd.concat([train_df, test_df])
    target = np.concatenate([train_y, test_y]).reshape(-1, 1)
    #

    model = RandomForestClassifier()
    model.fit(combined_df, target.ravel())

    pred = model.predict(combined_df)

    roc_scores = roc_auc_score(target, pred, average=None)

    plot_roc_curve(model, combined_df, target)
    plt.show()
    print(roc_scores, np.mean(roc_scores))
