"""
remedial computational applied math
computing square roots numerically
"""


def solve(a, x_0, eps):
    """show-your-working version"""
    # solve x**2 = a

    # let f(x) := x**2 - a
    # <=> solve f(x) = 0

    # let g(x) := f(x)**2
    # <=> min_x g(x)

    f = lambda z : z**2 - a
    df = lambda z : 2*z
    ddf = lambda z : 2
    # g = lambda z : f(z) ** 2 # unused
    dg = lambda z : 2 * f(z) * df(z)
    ddg = lambda z : 2 * df(z) ** 2 + 2 * f(z) * ddf(z)

    # gradient descent with variable step size - inversely proportional to norm of 2nd derivative
    tau = lambda z : 1.0 / abs(ddg(z))
    h = lambda z : z - tau(z) * dg(z)

    iters = 0
    x = x_0
    while not abs(f(x)) < eps:
        x = h(x)
        iters += 1
    return x, iters


def solve_tersely(a, x_0, eps):
    iters = 0
    z = x_0
    zz = z ** 2
    fz = zz - a
    while not abs(fz) < eps:
        z = z - (4. * z * fz) / abs(12 * zz - 4 * a)
        zz = z ** 2
        fz = zz - a
        iters += 1
    return z, iters


def sqrt(a, x_0=None, eps=None, worker=solve):
    if a < 0.0:
        raise ValueError(a) # NOPE
    return worker(a, x_0 or a / 2.0, eps or 1.0e-12)


a = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1] + range(1, 1000 + 1)
sqrt_a, iters = zip(*map(sqrt, a))
sqrt_a_, _ = zip(*[sqrt(ai, worker=solve_tersely) for ai in a])

import pylab
pylab.plot(a, sqrt_a, 'k-', label='$\sqrt{a}$')
pylab.plot(a, sqrt_a_, 'b-', label='$\sqrt{a}$')
pylab.plot(a, iters, 'r--', label='iters')
pylab.legend(loc='lower right')
pylab.xlabel('$a$')
pylab.show()

