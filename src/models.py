"""Custom survival Naive Bayes, following the reference paper's description.

For each time t (1..max TTE in training data):
  y = 1  -> event has NOT occurred by t (TTE > t),  prior p(y=1) = S(t) (Kaplan-Meier)
  y = 0  -> event HAS occurred by t   (TTE <= t),  prior p(y=0) = 1 - S(t)
  p(x_j | y) are Gaussians fit on the training set at that t.
The predicted TTE is the first t at which p(y=1 | x) drops below 0.5.
"""
import numpy as np


def kaplan_meier(times, max_t):
    """S(t) for t = 0..max_t. Every event is observed (no censoring)."""
    S = np.ones(max_t + 1)
    at_risk, s = len(times), 1.0
    for t in range(1, max_t + 1):
        d = int(np.sum(times == t))
        if at_risk > 0:
            s *= 1 - d / at_risk
        at_risk -= d
        S[t] = s
    return S


class SurvivalNaiveBayes:
    # var_smoothing is OUR choice (the reference does not state one):
    # a fraction of each feature's overall variance added to every class variance.
    def __init__(self, var_smoothing=1e-3, normalize_density=True):
        self.var_smoothing = var_smoothing
        # The reference code omits the 1/sigma factor of the Gaussian density.
        # normalize_density=False mimics that; True is the correct density.
        self.normalize_density = normalize_density

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.max_t = int(self.y.max())
        self.S = kaplan_meier(self.y, self.max_t)
        self.eps = self.var_smoothing * self.X.var(axis=0) + 1e-30
        return self

    def _loglik(self, x, mu, var):
        quad = np.sum((x - mu) ** 2 / var, axis=1)
        const = np.log(2 * np.pi) * x.shape[1]
        if self.normalize_density:
            const = const + np.sum(np.log(var))
        return -0.5 * (const + quad)

    def predict(self, Xnew):
        Xnew = np.asarray(Xnew, float)
        pred = np.full(len(Xnew), float(self.max_t))
        done = np.zeros(len(Xnew), dtype=bool)
        for t in range(1, self.max_t + 1):
            alive = self.y > t
            p1 = self.S[t]
            if alive.all() or not alive.any() or p1 <= 0 or p1 >= 1:
                continue
            X1, X0 = self.X[alive], self.X[~alive]
            ll1 = np.log(p1) + self._loglik(Xnew, X1.mean(0), X1.var(0) + self.eps)
            ll0 = np.log(1 - p1) + self._loglik(Xnew, X0.mean(0), X0.var(0) + self.eps)
            hit = (~done) & (ll0 > ll1)
            pred[hit] = t
            done |= hit
        return pred