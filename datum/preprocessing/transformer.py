import pandas as pd
import umap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE


class LowDimFeature(BaseEstimator, TransformerMixin):
    """
    Create features based on dimesionality reduction techniques, eg: umap, tsne
    """

    def __init__(self, n_components=2,
                 is_umap=True, is_tsne=True,
                 tsne_perplexity=10):
        """_summary_

        Args:
            n_components (int, optional): no of componests. Defaults to 2.
            is_umap (bool, optional): If True use umap for dimentionality reduction. Defaults to True.
            is_tsne (bool, optional):If True use umap for dimentionality reduction. Defaults to True.
            tsne_perplexity (int, optional):  Defaults to 10.
        """
        self.is_umap = is_umap
        self.is_tsne = is_tsne
        self.n_components = n_components
        self.reducer1 = umap.UMAP(n_components=n_components)
        self.reducer2 = TSNE(n_components=n_components,
                             learning_rate='auto',
                             init='random', perplexity=tsne_perplexity)

    def fit(self, X, y=None):
        if self.is_umap:
            self.reducer1.fit(X)
        if self.is_tsne:
            self.reducer2.fit(X)
        return self

    def transform(self, X, y=None):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        if self.is_umap:
            df1 = self.reducer1.transform(X)
            df1 = pd.DataFrame(df1, columns=[
                               f"umap{i+1}" for i in range(self.n_components)])
        if self.is_tsne:
            df2 = self.reducer2.transform(X)
            df2 = pd.DataFrame(df2, columns=[
                               f"tsne{i+1}" for i in range(self.n_components)])

        df = pd.concat([df1, df2], axis=1)
        return df
