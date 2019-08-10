import numpy as np
# from prml.random import Gaussian
from .HMM import HiddenMarkovModel


class RandomVariable(object):
    """
    base class for random variables
    """

    def __init__(self):
        self.parameter = {}

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            if isinstance(value, RandomVariable):
                string += f"{key}={value:8}"
            else:
                string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    def __format__(self, indent="4"):
        indent = int(indent)
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * indent)
            if isinstance(value, RandomVariable):
                string += f"{key}=" + value.__format__(str(indent + 4))
            else:
                string += f"{key}={value}"
            string += "\n"
        string += (" " * (indent - 4)) + ")"
        return string

    def fit(self, X, **kwargs):
        """
        estimate parameter(s) of the distribution

        Parameters
        ----------
        X : np.ndarray
            observed data
        """
        self._check_input(X)
        if hasattr(self, "_fit"):
            self._fit(X, **kwargs)
        else:
            raise NotImplementedError

    # def ml(self, X, **kwargs):
    #     """
    #     maximum likelihood estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_ml"):
    #         self._ml(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def map(self, X, **kwargs):
    #     """
    #     maximum a posteriori estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_map"):
    #         self._map(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def bayes(self, X, **kwargs):
    #     """
    #     bayesian estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_bayes"):
    #         self._bayes(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    def pdf(self, X):
        """
        compute probability density function
        p(X|parameter)

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(X)
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError

    def draw(self, sample_size=1):
        """
        draw samples from the distribution

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, ndim) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        assert isinstance(X, np.ndarray)

class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu, var)
    = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mu=None, var=None, tau=None):
        super().__init__()
        self.mu = mu
        if var is not None:
            self.var = var
        elif tau is not None:
            self.tau = tau
        else:
            self.var = None
            self.tau = None

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            self.parameter["mu"] = np.array(mu)
        elif isinstance(mu, np.ndarray):
            self.parameter["mu"] = mu
        elif isinstance(mu, Gaussian):
            self.parameter["mu"] = mu
        else:
            if mu is not None:
                raise TypeError(f"{type(mu)} is not supported for mu")
            self.parameter["mu"] = None

    @property
    def var(self):
        return self.parameter["var"]

    @var.setter
    def var(self, var):
        if isinstance(var, (int, float, np.number)):
            assert var > 0
            var = np.array(var)
            assert var.shape == self.shape
            self.parameter["var"] = var
            self.parameter["tau"] = 1 / var
        elif isinstance(var, np.ndarray):
            assert (var > 0).all()
            assert var.shape == self.shape
            self.parameter["var"] = var
            self.parameter["tau"] = 1 / var
        else:
            assert var is None
            self.parameter["var"] = None
            self.parameter["tau"] = None

    @property
    def tau(self):
        return self.parameter["tau"]

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            assert tau > 0
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, np.ndarray):
            assert (tau > 0).all()
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = 1 / tau
        elif isinstance(tau, Gamma):
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
            self.parameter["var"] = None
        else:
            assert tau is None
            self.parameter["tau"] = None
            self.parameter["var"] = None

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        elif mu_is_gaussian:
            self._bayes_mu(X)
        elif tau_is_gamma:
            self._bayes_tau(X)
        else:
            self._ml(X)

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def _map(self, X):
        assert isinstance(self.mu, Gaussian)
        assert isinstance(self.var, np.ndarray)
        N = len(X)
        mu = np.mean(X, 0)
        self.mu = (
            (self.tau * self.mu.mu + N * self.mu.tau * mu)
            / (N * self.mu.tau + self.tau)
        )

    def _bayes_mu(self, X):
        N = len(X)
        mu = np.mean(X, 0)
        tau = self.mu.tau + N * self.tau
        self.mu = Gaussian(
            mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
            tau=tau
        )

    def _bayes_tau(self, X):
        N = len(X)
        var = np.var(X, axis=0)
        a = self.tau.a + 0.5 * N
        b = self.tau.b + 0.5 * N * var
        self.tau = Gamma(a, b)

    def _bayes(self, X):
        N = len(X)
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        if mu_is_gaussian and not tau_is_gamma:
            mu = np.mean(X, 0)
            tau = self.mu.tau + N * self.tau
            self.mu = Gaussian(
                mu=(self.mu.mu * self.mu.tau + N * mu * self.tau) / tau,
                tau=tau
            )
        elif not mu_is_gaussian and tau_is_gamma:
            var = np.var(X, axis=0)
            a = self.tau.a + 0.5 * N
            b = self.tau.b + 0.5 * N * var
            self.tau = Gamma(a, b)
        elif mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _pdf(self, X):
        d = X - self.mu
        return (
            np.exp(-0.5 * self.tau * d ** 2) / np.sqrt(2 * np.pi * self.var)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self.mu,
            scale=np.sqrt(self.var),
            size=(sample_size,) + self.shape
        )



class GaussianHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with Gaussian emission model
    """

    def __init__(self, initial_proba, transition_proba, means, covs):
        """
        construct hidden markov model with Gaussian emission model

        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray or None
            probability of initial states
        transition_proba : (n_hidden, n_hidden) np.ndarray or None
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean of each gaussian component
        covs : (n_hidden, ndim, ndim) np.ndarray
            covariance matrix of each gaussian component

        Attributes
        ----------
        ndim : int
            dimensionality of observation space
        n_hidden : int
            number of hidden states
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0] == covs.shape[0]
        assert means.shape[1] == covs.shape[1] == covs.shape[2]
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means
        self.covs = covs
        self.precisions = np.linalg.inv(self.covs)
        self.gaussians = [Gaussian(m, cov) for m, cov in zip(means, covs)]

    def draw(self, n=100):
        """
        draw random sequence from this model

        Parameters
        ----------
        n : int
            length of the random sequence

        Returns
        -------
        seq : (n, ndim) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.extend(self.gaussians[hidden_state].draw())
#             seq.extend(np.random.normal(loc=self.means[hidden_state], scale=np.sqrt(self.cov[hidden_state])), size=(1,)+self.means.shape)
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self, X):
        diff = X[:, None, :] - self.means
        exponents = np.sum(
            np.einsum('nki,kij->nkj', diff, self.precisions) * diff, axis=-1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs) * (2 * np.pi) ** self.ndim)

    def maximize(self, seq, p_hidden, p_transition):
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        Nk = np.sum(p_hidden, axis=0)
        self.means = (seq.T @ p_hidden / Nk).T
        diffs = seq[:, None, :] - self.means
        self.covs = np.einsum('nki,nkj->kij', diffs, diffs * p_hidden[:, :, None]) / Nk[:, None, None]
