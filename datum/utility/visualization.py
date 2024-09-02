import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def plot_importance(model, feature_names, topk=10):
    # plot feature importance
    if isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in model.estimators_], axis=0)
        feat_importances = pd.Series(importances, index=feature_names)
        feat_importances = feat_importances.nlargest(topk)
        fig, ax = plt.subplots(figsize=(8, 5))
        feat_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("importance")
        fig.tight_layout()
    elif isinstance(model, LinearRegression) or isinstance(model, LogisticRegression):
        coefficients = model.coef_[0]
        # Create a dataframe for coefficients and sort by absolute value
        importance = pd.DataFrame(
            {'Feature': feature_names, 'Coefficient': coefficients})
        importance['AbsCoefficient'] = np.abs(importance['Coefficient'])
        importance = importance.sort_values(
            by='AbsCoefficient', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance['Feature'],
                 importance['AbsCoefficient'], color='skyblue')
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance in linear model')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.show()
