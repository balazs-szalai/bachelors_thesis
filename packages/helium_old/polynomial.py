import numbers
import numpy as np
from pdb import set_trace

class Polynomial(tuple):
    def __new__(self, arg):
        if isinstance(arg, Polynomial):
            return arg

        if isinstance(arg, numbers.Number):
            arg = (arg,)
        elif not isinstance(arg[0], np.ndarray):
            n = len(arg)
            while n > 0:
                if arg[n - 1]: break
                n -= 1

            if not n:
                return (0,)
            else:
                arg = arg[:n]

        return tuple.__new__(self, arg)

    def __getitem__(self, n):
        if n < len(self):
            return tuple.__getitem__(self, n)
        return 0

    def __repr__(self):
        return '(' + ', '.join(str(x) for x in self) + ')'

    def __add__(self, other):
        if other == (0,):
            return self

        other = Polynomial(other)

        n = max(len(self), len(other))
        return Polynomial([self[i] + other[i] for i in range(n)])

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Polynomial([-x for x in self])
    
    def __sub__(self, other):
        return self + -Polynomial(other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = Polynomial(other)
        res = [0]*(len(self) + len(other) - 1)
        for i,a in enumerate(self):
            for j,b in enumerate(other):
                res[i+j] += a*b
        return Polynomial(res)

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, x):
        return Polynomial([v/x for v in self])

    def __pow__(self, n):
        if n == 1:
            return self

        n1 = n//2
        half = self**n1

        res = half*half
        if n == n1*2:
            return res
        else:
            return res*self

    def evaluate(self, x):
        res = 0
        for p in reversed(self):
            res = res*x + p
        return res

    def diff(self):
        return Polynomial([i*x for i,x in enumerate(self)][1:])


