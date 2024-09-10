import matplotlib as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.cluster import hierarchy as hc


def get_hierchical_clusters(X, drop_na=True):
    if drop_na:
        X = X.dropna()
        
    corr = np.round(scipy.stats.spearmanr(X).correlation, 4)
    plt.figure(figsize=(16, 10))
    hc.dendrogram(hc.linkage(hc.distance.squareform(1-corr),
                             method='average'),
                  labels=X.columns, orientation='left',
                  leaf_font_size=16)
    plt.show()


def get_corr_plot(corr, title="correlations"):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(title, fontsize=14)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, False)

    # Generate the heatmap including the mask
    heatmap = sns.heatmap(corr,
                          annot=True,
                          annot_kws={"fontsize": 10},
                          fmt='.2f',
                          linewidths=0.5,
                          cmap='RdBu',
                          mask=mask,  # the mask has been included here
                          ax=ax)

    # Display our plot
    plt.show()


def dist_visualization(X_train, X_test, cont_cols):
    """Visualize distribution of train and test data (continious values)"""

    # Calculate the number of rows needed for the subplots
    num_rows = (len(cont_cols) + 2) // 3

    X = pd.concat([X_train, X_test], axis=0)
    # Create subplots for each continuous column
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

    # Loop through each continuous column and plot the histograms
    for i, col in enumerate(cont_cols):
        # Determine the range of values to plot
        max_val = max(X_train[col].max(), X_test[col].max(), X[col].max())
        min_val = min(X_train[col].min(), X_test[col].min(), X[col].min())
        range_val = max_val - min_val

        # Determine the bin size and number of bins
        bin_size = range_val / 20
        num_bins_train = round(range_val / bin_size)
        num_bins_test = round(range_val / bin_size)
        num_bins_original = round(range_val / bin_size)

        # Calculate the subplot position
        row = i // 3
        col_pos = i % 3

        # Plot the histograms
        sns.histplot(X_train[col], ax=axs[row][col_pos], color='orange',
                     kde=True, label='Train', bins=num_bins_train)
        sns.histplot(X_test[col], ax=axs[row][col_pos], color='green',
                     kde=True, label='Test', bins=num_bins_test)
        sns.histplot(X[col], ax=axs[row][col_pos], color='blue',
                     kde=True, label='Original', bins=num_bins_original)
        axs[row][col_pos].set_title(col)
        axs[row][col_pos].set_xlabel('Value')
        axs[row][col_pos].set_ylabel('Frequency')
        axs[row][col_pos].legend()

    # Remove any empty subplots
    if len(cont_cols) % 3 != 0:
        for col_pos in range(len(cont_cols) % 3, 3):
            axs[-1][col_pos].remove()

    plt.tight_layout()
    plt.show()
    
    
def plot_missing_values(df, title, color):
    missing_ratio = df.isnull().sum() / len(df) * 100
    missing_df = pd.DataFrame({'column': missing_ratio.index, 'missing_ratio': missing_ratio.values})
    
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    ax = sns.barplot(x='column', y='missing_ratio', data=missing_df, color=color)
    
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    
    plt.yticks(range(0, 101, 20))
    plt.ylabel('Missing Values ratio(%)')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.8,
                '{:.1f}%'.format(height),
                ha="center")
    
    plt.tight_layout()
    plt.show()
