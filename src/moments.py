import numpy as np

from util import Interval, clip, get_grid

def get_first_order_constraints(prob, mu_scalar):
    return clip(Interval(lo=(mu_scalar - 1 + prob) / prob,
                         hi=mu_scalar/prob))

def get_second_order_constraints(S, rho, mu_2dim, other_interval):
    other_grid = get_grid(other_interval)
    this_grid  = (S - rho - mu_2dim[1]*other_grid) / (mu_2dim[0] - other_grid)

    return clip(Interval(lo=this_grid.min(), hi=this_grid.max()))

def get_feasible_centers_brute(pivot, pi_hat, mu_hat, S_hat, Rho,
                               base_intervals):
    pivot_cluster, pivot_dim = pivot
    dim = mu_hat.shape[0]

    pi_hat_pivot = pi_hat[pivot_cluster]
    mu_hat_pivot = mu_hat[pivot_dim]

    pi_hat_other = pi_hat[1-pivot_cluster]

    def is_base_feasible(center, b, j):
        return (center >= base_intervals[b][j].lo and
                center <= base_intervals[b][j].hi)

    def other_center(center, prob, mean_hat):
        return (mean_hat - prob*center) / (1 - prob)

    pivot_interval = get_grid(get_first_order_constraints(pi_hat_pivot, mu_hat_pivot),
                              1000)
    feasible_solutions = []
    for pivot_center in pivot_interval:
        candidate_solution = np.zeros((2, dim))
        candidate_solution[pivot_cluster,pivot_dim] = pivot_center
        candidate_solution[1-pivot_cluster,pivot_dim] = other_center(pivot_center,
                                                                     pi_hat_pivot,
                                                                     mu_hat_pivot)

        feasible = True
        for j in range(dim):
            if j == pivot_dim: continue

            jth_center = ((S_hat[pivot_dim,j] - Rho[pivot_dim,j] - mu_hat[j]*pivot_center) /
                          (mu_hat_pivot - pivot_center))
            if not is_base_feasible(jth_center, 1-pivot_cluster, j):
                feasible = False
                break
            else:
                candidate_solution[1-pivot_cluster,j] = jth_center
                candidate_solution[pivot_cluster,j] = other_center(pivot_center,
                                                                   pi_hat_other,
                                                                   mu_hat[j])

        if feasible:
            feasible_solutions.append(candidate_solution)

    return feasible_solutions

def get_feasible_centers_cvx(pi_hat, mu_hat, S_hat, Rho, base_intervals):
    import cvxpy as cvx
    import qcqp
    
    dim = mu_hat.shape[0]

    centers = cvx.Variable(2, dim)
    print(centers[0,1].__class__)
    
    constraints = [
        mu_hat == centers.T * pi_hat,
        *[
            S_hat[i,j] == (Rho[i,j] + pi_hat[0] * centers[0,i] * centers[0,j]
                                    + pi_hat[1] * centers[1,i] * centers[1,j])
            for i in list(range(dim-1)) for j in list(range(i + 1, dim))
        ]
    ]

    obj = cvx.Minimize(1)
    problem = cvx.Problem(obj, constraints)
    problem.solve(method='qcqp-admm')
    
    if problem.status != cvx.OPTIMAL:
        raise Exception("cvx returned problem status {}".format(problem.status))
    return centers.value
