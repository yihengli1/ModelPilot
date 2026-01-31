from sklearn.decomposition import PCA
import numpy as np


# If u want to PCA through the application
class PCAModel:
    def __init__(self, n_components, svd_solver="auto", whiten=False, random_state=42):
        print("init pca", flush=True)
        self.pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten,
            random_state=random_state,
        )
        self.is_fit_ = False
        self.k_components_ = None
        self.variance_explained_ = None

    def _update_pca_stats(self):
        self.k_components_ = int(getattr(self.pca, "n_components_", 0) or 0)
        evr = getattr(self.pca, "explained_variance_ratio_", None)
        self.variance_explained_ = float(
            np.sum(evr)) if evr is not None else 0.0

    def fit(self, X, y=None):
        print("PCA fit", flush=True)
        self.pca.fit(X)
        self._update_pca_stats()
        self.is_fit_ = True
        return self

    def transform(self, X):
        if not self.is_fit_:
            raise ValueError("PCA is not fit yet.")
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        print("PCA fit_transform", flush=True)
        Z = self.pca.fit_transform(X)
        self._update_pca_stats()
        self.is_fit_ = True
        return Z


# preprocessing
class PCAPipeline:
    def __init__(
        self,
        model,
        n_components=None,
        svd_solver="auto",
        whiten=False,
        random_state=42,
    ):
        self.pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten,
            random_state=random_state,
        )
        self.model = model
        self.is_fit_ = False

        # Artifacts
        self.k_components_ = None
        self.variance_explained_ = None

    def _update_pca_stats(self) -> None:
        self.k_components_ = int(getattr(self.pca, "n_components_", 0) or 0)
        evr = getattr(self.pca, "explained_variance_ratio_", None)
        self.variance_explained_ = float(
            np.sum(evr)) if evr is not None else 0.0

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

    def transform(self, X):
        if not self.is_fit_:
            raise ValueError("Pipeline is not fit yet.")
        return self.pca.transform(X)

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
