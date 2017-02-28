import numpy as np

import util

class MixtureModel(object):
    def __init__(self, components, probs):
        self.K = len(components)
        self.components = components

        n_moments = 1000000
        self.means = []
        self.covs = []
        for k, component in enumerate(components):
            Xk = component.draw(n_moments)
            self.means.append(np.mean(Xk, axis=0))
            self.covs.append(util.cov(Xk))

        self.probs = probs

    def mean(self, k):
        assert k >= 0 and k < self.K
        return self.means[k]

    def cov(self, k):
        assert k >= 0 and k < self.K
        return self.covs[k]

    def second_moment(self, k):
        return (self.covs[k] + np.outer(self.means[k], self.means[k]))

    def draw(self, size):
        return np.array([
            self.components[np.random.choice(np.arange(self.K), p=self.probs)].draw(1)[0,:]
            for _ in range(size)
        ])

class Sampler(object):
    def __init__(self):
        pass

    def draw(self, size):
        raise Exception("Not implemented.")

class LatentGaussianSampler(Sampler):
    def __init__(self, mu, Sigma, obs=lambda _ : _):
        self.mu = mu
        self.Sigma = Sigma
        self.obs = obs

    def draw(self, size):
        return self.obs(np.random.multivariate_normal(self.mu, self.Sigma, size))

if __name__ == "__main__":
    mus   = np.array([[0.5, -0.5], [-0.5, 0.5]])
    Sigma = np.array([[1., 0.7], [0.7, 1.]])
    obs = util.threshold

    mixture = MixtureModel(
        [ LatentGaussianSampler(mus[0,:], Sigma, obs),
          LatentGaussianSampler(mus[1,:], Sigma, obs)
        ], probs = np.array([0.5, 0.5])
    )

    print(mixture.means)
    print()
    print(mixture.covs[0])
    print()
    print(mixture.covs[1])
    print()

    X = mixture.draw(10)
    print(X)