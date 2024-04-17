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
        self.is_umap = is_umap
        self.is_tsne = is_tsne
        self.n_components = n_components
        self.reducer1 = umap.UMAP(n_components=n_components)
        self.reducer2 = TSNE(n_components=n_components,
                             learning_rate='auto',
                             init='random', perplexity=tsne_perplexity)

    def fit(self, X):
        if self.is_umap:
            self.reducer1.fit(X)
        if self.is_tsne:
            self.reducer2.fit(X)
        return self

    def transform(self, X):
        if self.is_umap:
            transform1 = self.reducer1.transform(X)
            df1 = pd.DataFrame(transform1, columns=[
                               f"umap{i+1}" for i in range(self.n_components)])
        if self.is_tsne:
            transform2 = self.reducer2.transform(X)
            df2 = pd.DataFrame(transform2, columns=[
                               f"tsne{i+1}" for i in range(self.n_components)])

        if self.is_umap and self.is_tsne:
            df = pd.concat([transform1, transform2], axis=1)
            return df
        if self.is_umap:
            return df1
        if self.is_tsne:
            return df2
