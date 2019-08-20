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
    #raise NotImplementedError
    n, d = X.shape
    # compute posterior
    mu = mixture.mu
    var = mixture.var.reshape(-1, 1)
    p = mixture.p.reshape(-1, 1)

    # masking for non-complete vectors
    d_s = np.sum(X!=0, axis=1)
    #mask = X != 0
    #X = np.ma.masked_array(X, mask=mask)
    diff = np.where(X!=0, X - mu.reshape(-1, 1, d), 0)
    sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    #sse = np.exp(2 * np.log(np.linalg.norm(diff, ord=2, axis=2)))
    #lh = (1/(2*np.pi*var)**(d_s/2)) * np.exp((-1/(2*var))*sse)
    #lh = (1/np.exp(d_s/2 * np.log(2*np.pi*var))) * np.exp((-1/(2*var))*sse)
    #lh[lh==0] = 1**-16
    #print(np.sum(lh==0))
    #log_gauss = np.log(1/np.exp(d_s/2 * np.log(2*np.pi*var))+ 1e-16)-1/(2*var)*sse
    log_gauss = -(d_s/2) * np.log(2*np.pi*var) - (1/(2*var))*sse
    #post = np.log(p) + np.log(lh) - logsumexp(np.log(p*lh), axis=0)
    #post = np.log(p + 1e-16) + np.log(lh+ 1e-16) - logsumexp(np.log(p+ 1e-16)+np.log(lh+ 1e-16), axis=0)
    post = np.log(p + 1e-16) + log_gauss - logsumexp(np.log(p+ 1e-16)+log_gauss, axis=0)
    post = np.exp(post).T

    # diff = X - mu.reshape(-1, 1, d)
    # sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    # lh = (1/(2*np.pi*var)**(d/2)) * np.exp((-1/(2*var))*sse)

    # post = (p * lh)/(np.sum(p*lh, axis=0))
    # post = post.T


    #log_lh = np.log(np.sum(p * lh, axis=0)).sum()
    #log_lh = logsumexp(np.log(p+ 1e-16) + np.log(lh+ 1e-16), axis=0).sum()
    log_lh = logsumexp(np.log(p+ 1e-16) + log_gauss, axis=0).sum()

    return post, log_lh



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = 0.25) -> GaussianMixture:
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
    #raise NotImplementedError
    n, d = X.shape
    mu = mixture.mu

    d_s = np.sum(X!=0, axis=1)

    n_hat = np.sum(post, axis=0)
    p = n_hat / n

    # condition for mean update
    mask = np.int32(X!=0)
    condition = np.matmul(post.T, mask)

    #new_mu = np.matmul(post.T, X) / np.matmul(post.T, mask)
    #new_mu = np.matmul(post.T, X)/(condition+1e-16)
    new_mu = np.divide(np.matmul(post.T, X), condition, out=np.zeros_like(mu), where=condition!=0)
    mu[condition>=1] = new_mu[condition>=1]
    #mu = new_mu
    diff = np.where(X!=0, X - mu.reshape(-1, 1, d), 0)
    sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    #sse = np.exp(2*np.log(np.linalg.norm(diff, ord=2, axis=2)))
    new_var = (1/np.matmul(post.T, d_s)) * np.sum(post.T * sse, axis=1)

    # set a minimum varaince tp prevent variance from going to zero die to a 
    # small number of points being assigned to them
    new_var[new_var < min_variance] = min_variance
    # mu = np.matmul(post.T, X) / n_hat.reshape(-1, 1)

    # diff = X - mu.reshape(-1, 1, d)
    # sse = np.linalg.norm(diff, ord=2, axis=2)**2 # k by n
    # sse = sse.T
    # var = np.sum(post * sse, axis=0)/(n_hat * d)

    mixture = GaussianMixture(mu, new_var, p)

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
        mixture = mstep(X, post, mixture, min_variance=0.25)

        # convergence criteria
        #test = log_lh - prev_log_lh > 10e-6*np.abs(log_lh)

    return mixture, post, log_lh


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    #raise NotImplementedError
    #raise NotImplementedError
    n, d = X.shape
    # compute posterior
    mu = mixture.mu
    var = mixture.var.reshape(-1, 1)
    p = mixture.p.reshape(-1, 1)

    # masking for non-complete vectors
    d_s = np.sum(X!=0, axis=1)
    #mask = X != 0
    #X = np.ma.masked_array(X, mask=mask)
    diff = np.where(X!=0, X - mu.reshape(-1, 1, d), 0)
    sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    #sse = np.exp(2 * np.log(np.linalg.norm(diff, ord=2, axis=2)))
    #lh = (1/(2*np.pi*var)**(d_s/2)) * np.exp((-1/(2*var))*sse)
    #lh = (1/np.exp(d_s/2 * np.log(2*np.pi*var))) * np.exp((-1/(2*var))*sse)
    #lh[lh==0] = 1**-16
    #print(np.sum(lh==0))
    #log_gauss = np.log(1/np.exp(d_s/2 * np.log(2*np.pi*var))+ 1e-16)-1/(2*var)*sse
    log_gauss = -(d_s/2) * np.log(2*np.pi*var) - (1/(2*var))*sse
    #post = np.log(p) + np.log(lh) - logsumexp(np.log(p*lh), axis=0)
    #post = np.log(p + 1e-16) + np.log(lh+ 1e-16) - logsumexp(np.log(p+ 1e-16)+np.log(lh+ 1e-16), axis=0)
    post = np.log(p + 1e-16) + log_gauss - logsumexp(np.log(p+ 1e-16)+log_gauss, axis=0)
    post = np.exp(post).T

    # diff = X - mu.reshape(-1, 1, d)
    # sse = np.linalg.norm(diff, ord=2, axis=2) ** 2
    # lh = (1/(2*np.pi*var)**(d/2)) * np.exp((-1/(2*var))*sse)

    # post = (p * lh)/(np.sum(p*lh, axis=0))
    # post = post.T


    #log_lh = np.log(np.sum(p * lh, axis=0)).sum()
    #log_lh = logsumexp(np.log(p+ 1e-16) + np.log(lh+ 1e-16), axis=0).sum()

    # get the Expectation for each data point calculated as
    # sum over all all clusters, the product of the probability of being in the cluster (p(j|i))
    # and the mean (mu_j) of the cluster
    expectation = np.matmul(post, mixture.mu)
    X_hat = X.copy()
    X_hat[X==0] = expectation[X==0]

    return X_hat


