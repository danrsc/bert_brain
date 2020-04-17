import numpy as np

# based on
# https://github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/min_norm_solvers_numpy.py


__all__ = ['find_min_norm_element']


def _min_norm_pairwise(dot_products):
    x_dot_x = np.diag(dot_products)
    y_dot_y = np.expand_dims(x_dot_x, 0)
    x_dot_x = np.expand_dims(x_dot_x, 1)
    denominator = x_dot_x + y_dot_y - 2 * dot_products
    gamma = np.divide(-1.0 * (dot_products - y_dot_y), denominator, where=denominator != 0)
    cost = y_dot_y + gamma * (dot_products - y_dot_y)
    case_1 = dot_products >= x_dot_x
    case_2 = np.logical_and(dot_products >= y_dot_y, np.logical_not(case_1))
    gamma = np.where(case_2, 0.001, np.where(case_1, 0.999, gamma))
    cost = np.where(case_2, y_dot_y, np.where(case_1, x_dot_x, cost))
    return gamma, cost


def _next_point(current_solution, grad):
    proj_grad = grad - np.sum(grad) / len(current_solution)
    tm1 = -1.0 * current_solution[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - current_solution[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    t = 1
    if len(tm1[tm1 > 1e-7]) > 0:
        t = np.min(tm1[tm1 > 1e-7])
    if len(tm2[tm2 > 1e-7]) > 0:
        t = min(t, np.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + current_solution
    return _projection_to_simplex(next_point)


def _projection_to_simplex(v):
    idx_sort = np.argsort(-v)
    u = v[idx_sort]
    cssv = np.cumsum(u) - 1
    ind = np.arange(len(v)) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def find_min_norm_element(gradients, l2_norm=False, max_iter=256, tol=1e-6):
    shape = gradients[0].shape
    gradients = np.array(list(np.reshape(g.numpy(), -1) for g in gradients))
    # ignore 0 gradients
    norm = np.linalg.norm(gradients, axis=1)
    if not np.any(norm > 0):
        return np.ones(len(norm)) / len(norm), 0, np.reshape(gradients[0], shape)
    gradients = gradients[norm > 0]
    if len(gradients) == 1:
        solution = np.zeros_like(norm)
        solution[norm > 0] = 1
        return solution, norm[norm > 0], np.reshape(gradients[0], shape)
    if l2_norm:
        gradients_ = gradients / np.expand_dims(norm[norm > 0], 1)
    else:
        gradients_ = gradients
    dot_products = gradients_ @ gradients_.T
    gamma, cost = _min_norm_pairwise(dot_products)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    solution = np.zeros(len(dot_products), dtype=gamma.dtype)
    solution[i] = gamma[i, j]
    solution[j] = 1 - gamma[i, j]
    if len(solution) == 2:
        solution_ = np.zeros_like(norm)
        solution_[norm > 0] = solution
        return solution_, cost[i, j], np.reshape(np.sum(np.expand_dims(solution, 1) * gradients, axis=0), shape)
    for _ in range(max_iter):
        grad_dir = -1 * np.dot(dot_products, solution)
        new_point = _next_point(solution, grad_dir)
        v1v1 = np.sum(np.expand_dims(solution, 1) * np.expand_dims(solution, 0) * dot_products)
        v1v2 = np.sum(np.expand_dims(solution, 1) * np.expand_dims(new_point, 0) * dot_products)
        v2v2 = np.sum(np.expand_dims(new_point, 1) * np.expand_dims(new_point, 0) * dot_products)
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
        elif v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
        else:
            gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
            cost = v2v2 + gamma * (v1v2 - v2v2)
        new_solution = gamma * solution + (1 - gamma) * new_point
        change = new_solution - solution
        if np.sum(np.abs(change)) < tol:
            solution_ = np.zeros_like(norm)
            solution_[norm > 0] = new_solution
            return solution_, cost, np.reshape(np.sum(np.expand_dims(new_solution, 1) * gradients, axis=0), shape)
        solution = new_solution
    solution_ = np.zeros_like(norm)
    solution_[norm > 0] = solution
    return solution_, cost, np.reshape(np.sum(np.expand_dims(solution, 1) * gradients, axis=0), shape)
