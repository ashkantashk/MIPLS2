"""
This is a script used for testing and development. For JAX LOOCV in the JAX PLS algorithms, use the loocv method of the JAX algorithms instead.
"""

import jax.numpy as jnp
import jax
from tqdm import tqdm
from functools import partial
import numpy as np

def loocv(estimator, A, X_train_val, Y_train_val, metric_function, metric_names):
    all_indices = jnp.arange(X_train_val.shape[0])
    X_train_val = jnp.array(X_train_val, dtype=jnp.float64)
    Y_train_val = jnp.array(Y_train_val, dtype=jnp.float64)
    metric_value_lists = [[] for _ in metric_names]

    i = 0 # The first iLet's do the first iteration outside tqdm's timing estimate
    train_indices = jnp.nonzero(all_indices != i, size=X_train_val.shape[0] - 1)[0]
    X_train = jnp.take(X_train_val, train_indices, axis=0)
    Y_train = jnp.take(Y_train_val, train_indices, axis=0)
    X_val = jnp.take(X_train_val, jnp.array([i]), axis=0)
    Y_val = jnp.take(Y_train_val, jnp.array([i]), axis=0)
    estimator.fit(X_train, Y_train, A)
    Y_pred = estimator.predict(X_val)
    score = metric_function(Y_val, Y_pred)
    metric_values = score
    for j, m in enumerate(metric_values):
        metric_value_lists[j].append(m)
        
    for i in tqdm(range(1, X_train_val.shape[0])):
        train_indices = jnp.nonzero(all_indices != i, size=X_train_val.shape[0] - 1)[0]
        X_train = jnp.take(X_train_val, train_indices, axis=0)
        Y_train = jnp.take(Y_train_val, train_indices, axis=0)
        X_val = jnp.take(X_train_val, jnp.array([i]), axis=0)
        Y_val = jnp.take(Y_train_val, jnp.array([i]), axis=0)
        estimator.fit(X_train, Y_train, A)
        Y_pred = estimator.predict(X_val)
        score = metric_function(Y_val, Y_pred)
        metric_values = score
        for j, m in enumerate(metric_values):
            metric_value_lists[j].append(m)
    metrics = {}
    for name, lst in zip(metric_names, metric_value_lists):
        metrics[name] = np.asarray(lst)
    return metrics