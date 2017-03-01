import numpy as np

import moments
import util
from mixtures import MixtureModel, LatentGaussianSampler


def main():
    mus   = np.array([[0.5, -0.5], [-0.5, 0.5]])
    Sigma = np.array([[1., 0.7], [0.7, 1.]])
    obs = util.threshold

    pi = np.array([0.7, 0.3])

    mixture = MixtureModel(
        [ LatentGaussianSampler(mus[0,:], Sigma, obs),
          LatentGaussianSampler(mus[1,:], Sigma, obs)
        ], probs = pi)

    n = 10000
    X = mixture.draw(n)

    mu_hat = np.mean(X, axis=0) #pi[0] * mixture.mean(0) + pi[1] * mixture.mean(1)
    S_hat  = util.second_moment(X) #pi[0] * mixture.second_moment(0) + pi[1] * mixture.second_moment(1)
    Rho = pi[0] * mixture.cov(0) + pi[1] * mixture.cov(1)
    rho = 1.0
    Rho[0,1] = rho; Rho[1,0] = Rho[0,1]

    print("means =\n{}".format(mixture.means))
    print("mu_hat =\n{}".format(mu_hat))
    print("S_hat =\n{}".format(S_hat))
    print("rho = {}".format(rho))

    # First order intervals for the first cluster component.
    first_intervals = [
        [moments.get_first_order_constraints(pi[0], mu_hat[0]),
         moments.get_first_order_constraints(pi[0], mu_hat[1])],
        [moments.get_first_order_constraints(pi[1], mu_hat[0]),
         moments.get_first_order_constraints(pi[1], mu_hat[1])]
    ]

    eps = 0.1
    fake_first_intervals = [
        [util.Interval(lo=mixture.mean(0)[0] - eps, hi=mixture.mean(0)[0] + eps),
         util.Interval(lo=mixture.mean(0)[1] - eps, hi=mixture.mean(0)[1] + eps)],
        [util.Interval(lo=mixture.mean(1)[0] - eps, hi=mixture.mean(1)[0] + eps),
         util.Interval(lo=mixture.mean(1)[1] - eps, hi=mixture.mean(1)[1] + eps)]
    ]

    first_intervals = first_intervals
    base_intervals = first_intervals

    feasible_centers = moments.get_feasible_centers_cvx(pi, mu_hat, S_hat, Rho, base_intervals)

    #moments.get_feasible_centers_brute((0, 0),
    #                                   pi, mu_hat, S_hat, Rho, base_intervals)

    for centers in feasible_centers:
        print(centers[0], centers[1])

    for t in range(1):
        second_intervals = [
            [moments.get_second_order_constraints(S_hat[0, 1], rho, mu_hat[::-1],
                                                  base_intervals[1][1]),
             moments.get_second_order_constraints(S_hat[0, 1], rho, mu_hat,
                                                  base_intervals[1][0])],
            [moments.get_second_order_constraints(S_hat[0, 1], rho, mu_hat[::-1],
                                                  base_intervals[0][1]),
             moments.get_second_order_constraints(S_hat[0, 1], rho, mu_hat,
                                                  base_intervals[0][0])]
        ]

    #    base_intervals = [[util.intersect([interval, base_intervals[b][j]])
    #                        for j, interval in enumerate(second_intervals[b])]
    #                      for b in range(2)]
    #
    print()
    for b in [0]:
        for j in [0]:
            print("Cluster {}, dimension {}:".format(b + 1, j +1))
            print("First order interval: [{}, {}]".format(
                first_intervals[b][j].lo, first_intervals[b][j].hi
            ))
            print("Second order interval: [{}, {}]".format(
                second_intervals[b][j].lo, second_intervals[b][j].hi
            ))
            print()

if __name__ == "__main__":
    main()