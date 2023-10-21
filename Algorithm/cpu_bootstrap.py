import numpy as np
from typing import Union
from numpy.random import SeedSequence, default_rng


def bootstrap_index_generator(train_size: int, val_size: int, num_iters: int, seed: Union[int, None] = None):
    """
    A generator yielding index splits of (train, val) for use in SKLearn cross validation functions.
    """
    ss = SeedSequence(seed)
    rng = default_rng(ss)
    np.random.seed
    val_idxs = np.arange(train_size, train_size + val_size)
    for _ in range(num_iters):
        yield rng.choice(train_size, size=(train_size,)), val_idxs
