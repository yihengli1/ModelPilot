from sklearn.decomposition import PCA


# Use a wrapper class for pipilene
class PCAPipeline:
    def __init__(self, pca: PCA, model):
        self.pca = pca
        self.model = model
        self.is_fit_ = False

    def fit(self, X, y=None):
        Z = self.pca.fit_transform(X)
        if y is None:
            if hasattr(self.model, "fit"):
                self.model.fit(Z)
        else:
            self.model.fit(Z, y)
        self.is_fit_ = True
        return self

    def predict(self, X):
        if not self.is_fit_:
            raise ValueError("Pipeline is not fit yet.")
        Z = self.pca.transform(X)
        return self.model.predict(Z)

    def fit_predict(self, X):
        Z = self.pca.fit_transform(X)
        if hasattr(self.model, "fit_predict"):
            out = self.model.fit_predict(Z)
        else:
            self.model.fit(Z)
            out = self.model.predict(Z)
        self.is_fit_ = True
        return out
