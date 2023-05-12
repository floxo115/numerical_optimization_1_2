import random

import numpy as np

from newton_method_mod import NewtonMethodWithModifiedHessian


def test_make_pos_def():
    nm = NewtonMethodWithModifiedHessian(None)
    for i in range(100):
        size = random.randint(1,20)
        mul = np.random.randint(-100, 100, (size, size))
        B = np.random.rand(size, size) * mul

        output = nm.make_pos_def(B)

        is_pos_def = np.all(np.linalg.eigvals(output) > 0)
        assert is_pos_def
