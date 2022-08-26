"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mu, var, p = mixture
    n, d = X.shape
    K, _ = mu.shape

    def normal(x, mu, var):
        # d = x.shape[0]
        # return (1 / (2 * np.pi * var) ** (d/2)) * \
        #     (np.exp(- np.sum(np.square(x - mu)) / (2 * var)))
        d = len(x)
        log_prob = -d / 2.0 * np.log(2 * np.pi * var)
        log_prob -= 0.5 * ((x - mu) ** 2).sum() / var
        return log_prob

    ll = 0
    post = np.zeros((n, K))
    for u in range(n):
        f = np.zeros(K)  # specific for each user
        msk = X[u] != 0  # mask for each user
        for j in range(K):
            f[j] = np.log(p[j] + 1e-16) + \
                normal(X[u, msk], mu[j, msk], var[j])
        otherl = logsumexp(f)
        # based on given relations
        # l = f - otherl
        # post[u] = np.exp(l)
        post[u] = f - otherl
        # basically if you reverse engineer the way we got ll in naive_e, then it is clear why this is same
        ll += otherl

    return np.exp(post), ll


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K, _ = mixture.mu.shape

    mu, var, p = mixture.mu.copy(), np.zeros(K), post.sum(axis=0) / n
    # j is k from formulas 
    for j in range(K):
        num, den = 0, 0
        for l in range(d):
            msk = X[:, l] != 0
            nsum = np.sum(post[msk, j])
            if nsum >= 1: # only update if sum if >= 1
                mu[j, l] = (X[msk, l] @ post[msk, j]) / nsum
            num += ((mu[j, l] - X[msk, l]) ** 2) @ post[msk, j]
            den += nsum
        var[j] = num / den
    var = np.maximum(var, min_variance)

    return GaussianMixture(mu, var, p)

    # Mu can be updated like this
    mu = (post.T@X)/(post.T@(X != 0))
    # Var can be updated like this
    


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll, ll = None, None
    while prev_ll is None or ((ll - prev_ll) > 1e-6 * np.abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    def normal(x, mu, var):
        # d = x.shape[0]
        # return (1 / (2 * np.pi * var) ** (d/2)) * \
        #     (np.exp(- np.sum(np.square(x - mu)) / (2 * var)))
        d = len(x)
        return -d / 2.0 * np.log(2 * np.pi * var) - 0.5 * np.sum((x - mu) ** 2) / var

    mu, var, p = mixture
    n, d = X.shape
    K, _ = mu.shape

    X_pred = X.copy()
    for i in range(n):
        msknot0 = X[i, :] != 0
        # nmsk = ~msk
        post = np.zeros(K)
        for j in range(K):
            ll = normal(X[i, msknot0], mu[j, msknot0], var[j])
            post[j] = np.log(p[j]) + ll
        post = np.exp(post - logsumexp(post))
        X_pred[i, ~msknot0] = np.dot(post, mu[:, ~msknot0])

    return X_pred