import math

def select_reed_muller_params_poisson(n, p):
    lambda_val = n * p

    sigma = math.sqrt(lambda_val)

    T = math.ceil(lambda_val + sigma)

    m = int(math.log2(n))

    for r in range(m, -1, -1):
        t = 2 ** (m - r - 1) - 1
        if t >= T:
            return r, m

    return None

