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
    n, d = X.shape
    # compute posterior
    mu = mixture.mu
    var = mixture.var.reshape(-1, 1)
    p = mixture.p.reshape(-1, 1)

    diff = X - mu.reshape(-1, 1, d)
    sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    lh = (1/(2*np.pi*var)**(d/2)) * np.exp((-1/(2*var))*sse)

    post = (p * lh)/(np.sum(p*lh, axis=0))
    post = post.T


    log_lh = np.log(np.sum(p * lh, axis=0)).sum()

    return post, log_lh
    #raise NotImplementedError



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
    #raise NotImplementedError
    n, d = X.shape

    n_hat = np.sum(post, axis=0)
    p = n_hat / n
    mu = np.matmul(post.T, X) / n_hat.reshape(-1, 1)

    diff = X - mu.reshape(-1, 1, d)
    sse = np.linalg.norm(diff, ord=2, axis=2)**2 # k by n
    sse = sse.T
    var = np.sum(post * sse, axis=0)/(n_hat * d)

    mixture = GaussianMixture(mu, var, p)

    return mixture
    



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
    #raise NotImplementedError
    prev_log_lh = None
    log_lh = None

    while ((prev_log_lh is None) or (log_lh - prev_log_lh > 10e-7*np.abs(log_lh))):
        prev_log_lh = log_lh

        post, log_lh = estep(X, mixture)
        mixture = mstep(X, post)

        # convergence criteria
        #test = log_lh - prev_log_lh > 10e-6*np.abs(log_lh)

    return mixture, post, log_lh
