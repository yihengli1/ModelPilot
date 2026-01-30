from sklearn.decomposition import PCA


# Use a wrapper class for pipilene
class PCAPipeline:
    def __init__(self, model, n_components, svd_solver, whiten):
        self.pca = PCA(n_components=n_components,
                       svd_solver=svd_solver, whiten=whiten)
        self.model = model
        self.is_fit_ = False

        # Artifacts
        self.k_components_ = None
        self.variance_explained_ = None

    def _update_pca_stats(self):
        self.k_components_ = int(self.pca.n_components_)
        self.variance_explained_ = float(
            self.pca.explained_variance_ratio_.sum())

    def fit(self, X, y=None):
        Z = self.pca.fit_transform(X)
        self._update_pca_stats()

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
        self._update_pca_stats()
        if hasattr(self.model, "fit_predict"):
            out = self.model.fit_predict(Z)
        else:
            self.model.fit(Z)
            out = self.model.predict(Z)
        self.is_fit_ = True
        return out
