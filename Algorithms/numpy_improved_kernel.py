import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from sklearn.base import BaseEstimator


class PLS(BaseEstimator):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/%28SICI%291099-128X%28199701%2911%3A1%3C73%3A%3AAID-CEM435%3E3.0.CO%3B2-%23?casa_token=vTCKHNzCM0wAAAAA:dTwxubegJ2dDQrLeRaPcoJncEvn9jky5LDmXcOVC1kcNvo1WCG0QmJXnGbvM6sBnKy7wtn-fG8qc7snrDQ

    Parameters:
    algorithm: Whether to use algorithm #1 or #2. Defaults to #1.
    dtype: The float datatype to use in computation of the PLS algorithm. Defaults to numpy.float64. Using a lower precision will yield significantly worse results when using an increasing number of components due to propagation of numerical errors.
    """

    def __init__(self, algorithm: int = 1, dtype: np.float_ = np.float64) -> None:
        self.algorithm = algorithm
        self.dtype = dtype
        self.name = 'PLS'
        self.A = self.K = self.M = self.N = None

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model

        Returns:
        None

        Sets:
        self.B: PLS regression coefficients matrix (A x K x M)
        self.W: PLS weights matrix for X (K x A)
        self.P: PLS loadings matrix for X (K x A)
        self.Q: PLS Loadings matrix for Y (M x A)
        self.R: PLS weights matrix to compute scores T directly from original X (K x A)
        if algorithm is 1, then also sets self.T which is a PLS scores matrix of X (N x A)

        Note that the internal representation of all matrices (except self.B) are transposed for optimization purposes.
        """
        X = np.array(X, dtype=self.dtype)
        Y = np.array(Y, dtype=self.dtype)

        N, K = X.shape
        M = Y.shape[1]

        self.B = np.zeros(shape=(A, K, M), dtype=self.dtype)
        self.W = np.empty(shape=(A, K), dtype=self.dtype)
        self.P = np.empty(shape=(A, K), dtype=self.dtype)
        self.Q = np.empty(shape=(A, M), dtype=self.dtype)
        self.R = np.empty(shape=(A, K), dtype=self.dtype)
        if self.algorithm == 1:
            self.T = np.empty(shape=(A, N), dtype=self.dtype)
        self.A = A
        self.N = N
        self.K = K
        self.M = M

        # step 1
        XTY = X.T @ Y

        # Used for algorithm #2
        if self.algorithm == 2:
            XTX = X.T @ X

        for i in range(A):
            # step 2
            if M == 1:
                norm = la.norm(XTY)
                if np.isclose(norm, 0):
                    print(
                        f'Weight is close to zero. Stopping fitting after A = {i} component(s).')
                    break
                w = XTY / norm
            else:
                if M < K:
                    XTYTXTY = XTY.T @ XTY
                    eig_vals, eig_vecs = la.eigh(XTYTXTY)
                    q = eig_vecs[:, -1:]
                    q = q.reshape(-1, 1)
                    w = XTY @ q
                    w = w / la.norm(w)
                else:
                    XTYYTX = XTY @ XTY.T
                    eig_vals, eig_vecs = la.eigh(XTYYTX)
                    w = eig_vecs[:, -1:]
            self.W[i] = w.squeeze()

            # step 3
            r = np.copy(w)
            for j in range(i):
                r = r - self.P[j].reshape(-1, 1).T @ w * \
                    self.R[j].reshape(-1, 1)
            self.R[i] = r.squeeze()

            # step 4
            if self.algorithm == 1:
                t = X @ r
                self.T[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ X).T / tTt
            elif self.algorithm == 2:
                rXTX = r.T @ XTX
                tTt = rXTX @ r
                p = (rXTX.T / tTt)
            q = (r.T @ XTY).T / tTt
            self.P[i] = p.squeeze()
            self.Q[i] = q.squeeze()

            # step 5
            XTY = XTY - (p @ q.T) * tTt

            # compute regression coefficients
            self.B[i] = self.B[i-1] + r @ q.T

    def predict(self, X: npt.ArrayLike, A: None | int = None) -> npt.NDArray[np.float_]:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        A: Integer number of components to use in the prediction or None. If None, return the predictions for every component. Defaults to the maximum number of components, the model was fitted with.

        Returns:
        Y_hat: Predicted response variables matrix (N x M) or (A x N x M)
        """
        X = np.array(X, dtype=self.dtype)
        if A is None:
            return (X @ self.B)
        return X @ self.B[A-1]
