import tensorflow as tf

class PLS():
    """
    TensorFlow implementation partial least-squares using Improved Kernel PLS Algorithm # 1 by Dayal and MacGregor: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/%28SICI%291099-128X%28199701%2911%3A1%3C73%3A%3AAID-CEM435%3E3.0.CO%3B2-%23?casa_token=vTCKHNzCM0wAAAAA:dTwxubegJ2dDQrLeRaPcoJncEvn9jky5LDmXcOVC1kcNvo1WCG0QmJXnGbvM6sBnKy7wtn-fG8qc7snrDQ
    This implementation assumes that all relevant pre processing has been applied to predictor variables and target variables.
    """

    def __init__(self) -> None:
        with tf.device('GPU'):
            self.B = None
            self.W = None
            self.P = None
            self.Q = None
            self.R = None
            self.T = None
            self.i = None
            self.j = None
            self.r = None
            self.A = None
            self.N = None
            self.K = None
            self.M = None

    # def helper(self, M, K, i, j, r, A, X, XT, XTY, B, W, P, Q, R, T):
    #@tf.function#(jit_compile=True)
    def helper(self, X, XTY):
        # steps 2-5
        while tf.less(self.i, self.A):
            # step 2
            XTYT = tf.transpose(XTY) # Used for later computation
            if self.M == 1:
                norm = tf.linalg.norm(XTY)
                if tf.experimental.numpy.isclose(norm, tf.constant(0)):
                    tf.print(
                        'Weight is close to zero. Stopping fitting at A =', self.i, 'component(s).')
                    w = XTY
                    break
                else:
                    w = XTY / norm
            else:
                if self.M < self.K:
                    XTYTXTY = tf.matmul(XTYT, XTY)
                    eig_vals, eig_vecs = tf.linalg.eigh(XTYTXTY)
                    q = eig_vecs[:, -1]
                    q = tf.reshape(q, shape=(-1, 1))
                    XTYq = tf.matmul(XTY, q)
                    w = XTYq / tf.linalg.norm(XTYq)
                else:
                    XTYYTX = tf.matmul(XTY, tf.transpose(XTY))
                    #XTYYTX = tf.matmul(tf.matmul(Y, XTYT), X) # define YTX in the beginning and use it here.
                    eig_vals, eig_vecs = tf.linalg.eigh(XTYYTX)
                    w = tf.expand_dims(eig_vecs[:, -1], axis=-1)
            self.W[self.i].assign(w)

            # step 3
            self.r.assign(tf.identity(w))
            self.j.assign(0)
            while tf.less(self.j, self.i):
                self.r.assign(self.r - tf.matmul(tf.transpose(self.P[self.j]), w) * self.R[self.j])
                # Increment inner loop counter
                self.j.assign_add(1)
            self.R[self.i].assign(self.r)

            # step 4
            t = tf.matmul(X, self.r)
            self.T[self.i].assign(t)
            tT = tf.transpose(t)
            tTt = tf.matmul(tT, t)
            p = tf.transpose(tf.matmul(tT, X)) / tTt
            q = tf.matmul(XTYT, self.r) / tTt
            self.P[self.i].assign(p)
            self.Q[self.i].assign(q)

            # step 5
            XTY.assign_add(- (tf.matmul(p, tf.transpose(q))) * tTt)

            # compute regression coefficients
            self.B[self.i].assign(tf.matmul(tf.transpose(tf.squeeze(
                self.R[:self.i+1], axis=-1)), tf.squeeze(self.Q[:self.i+1], axis=-1)))

            # Increment outer loop counter
            self.i.assign_add(1)

        return

    @tf.function#(jit_compile=True)
    def _get_initial_tensors(self, A, N, K, M, dtype):
        B = tf.zeros(shape=(A, K, M), dtype=dtype)
        W = tf.zeros(shape=(A, K, 1), dtype=dtype)
        P = tf.zeros(shape=(A, K, 1), dtype=dtype)
        Q = tf.zeros(shape=(A, M, 1), dtype=dtype)
        R = tf.zeros(shape=(A, K, 1), dtype=dtype)
        T = tf.zeros(shape=(A, N, 1), dtype=dtype)
        i = tf.convert_to_tensor(0, dtype=tf.int64)
        j = tf.convert_to_tensor(0, dtype=tf.int64)
        r = tf.identity(W[0])
        return B, W, P, Q, R, T, i, j, r

    @tf.function
    def _get_initial_XTY(self, X, Y):
        return tf.matmul(tf.transpose(X), Y)

    def fit(self, X, Y, A):
        """
        Parameters:
        X: Predictor variables matrix (N x K). The type of X will be used in all tensor operations.
        Y: Response variables matrix (N x M).
        A: Number of components in the PLS model.
        algorithm: Either 1 or 2. Whether to use algorithm #1 or algorithm #2.


        Returns:
        B: PLS regression coefficients matrix (A x K x M).
        W: PLS weights matrix for X (K x A).
        P: PLS loadings matrix for X (K x A).
        Q: PLS Loadings matrix for Y (M x A).
        R: PLS weights matrix to compute scores T directly from original X (K x A).
        T: PLS scores matrix of X (N x A)

        Note that the internal representations of all matrices (except B) are transposed and have an added dimension for optimization purposes.
        """

        dtype = X.dtype
        N, K = X.shape
        M = Y.shape[1]

        A = tf.convert_to_tensor(A, dtype=tf.int64)
        if A != self.A or N != self.N or K != self.K or M != self.M or dtype != self.dtype:
            self.A = A
            self.N = N
            self.K = K
            self.M = M
            self.dtype = dtype

            self.B, self.W, self.P, self.Q, self.R, self.T, self.i, self.j, self.r = map(tf.Variable, self._get_initial_tensors(A, N, K, M, dtype))

        # tf.print(f'Fitting a PLS model with {A} component(s) to map {N} data point(s) with {K} feature(s) to {N} target value(s) with {M} feature(s)')

        # step 1
        #XT = self._transpose_X(X)
        XTY = tf.Variable(self._get_initial_XTY(X, Y))
        # steps 2-5
        # return self.helper(M, K, i, j, r, A, X, XT, XTY, B, W, P, Q, R, T)
        return self.helper(X, XTY)

    @tf.function
    def predict(self, X, i):
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        i: Number of PLS components to use in the prediction. Integer between 1 and A.

        Returns:
        Y_hat: Predicted response variables matrix (N x K)
        """
        Y_hat = tf.matmul(X, self.B[i-1])
        return Y_hat