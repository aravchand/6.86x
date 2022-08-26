"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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
        return (1 / (2 * np.pi * var) ** (d/2)) * (np.exp(- np.sum(np.square(x - mu)) / (2 * var) ))
    numerator = np.zeros((n, K))
    # # Can optimize this by using the fact that the variance is the same for all components (btw this dont work)
    # for j in range(K):
    #     numerator[:, j] = p[j] * normal(X, mu[j], var[j])
        
    # denominator = np.sum(numerator, axis=1)
    # post = numerator / denominator[:, np.newaxis]
    # log_likelihood = np.sum(np.log(denominator))
    # return post, log_likelihood

    for i in range(n):
        for j in range(K):
            numerator[i][j] = p[j] * normal(X[i], mu[j], var[j]) 
    denominator = np.sum(numerator, axis=1) 
    post = numerator / denominator.reshape(-1, 1) 
    ll = float(np.sum(np.log(denominator)))
    return post, ll
    
    # var 
    # p = [p1. p2. p3. p4. ... p]

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    mu, var, p = np.zeros((K, d)), np.zeros((K, )), np.zeros((K, ))
    
    nsum = np.sum(post, axis=0)
    p = nsum / n
    # i feel like there are more efficient ways to do this
    for j in range(K):
        # post[:, j] = is the soft counts for the jth component
        # it is a vector in shape (n, )
        # so we make a vector in shape (n, 1) by adding a dimension using [:, np.newaxis]
        mu[j] = np.sum(X * post[:, j][:, np.newaxis], axis=0) / nsum[j]
        var[j] = np.sum((mu[j] - X) ** 2, axis=1) @ post[:, j] / (d * nsum[j])

    return GaussianMixture(mu, var, p)

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
    # mixture, post, ll = None, None, None
    prev_ll, ll = None, None
    while prev_ll is None or ((ll - prev_ll) > 1e-6 * np.abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, ll