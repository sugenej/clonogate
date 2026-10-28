"""v3.3: v3 with a non-uniform multiplicity prior pi_m ~ THETA**(m-1) over allowed m in 1..min(major,4).

UNATTESTED research model.  THETA=1 reproduces v3.  THETA<1 favours single-copy mutations (the usual case for subclonal SNVs).
Set env PARTITION_THETA before import.
"""
import os
import numpy as np

import partition_model_v3 as base

THETA = float(os.environ.get('PARTITION_THETA', '0.5'))
_orig_init = base._Data.__init__


def _init(self, a, n, major, minor, purity):
    _orig_init(self, a, n, major, minor, purity)
    ms = np.arange(1, self.mmax + 1)
    allowed = ms[None, :] <= self.M[:, None]
    w = np.where(allowed, THETA ** (ms[None, :] - 1.0), 0.0)
    w = w / w.sum(1, keepdims=True)
    with np.errstate(divide='ignore'):
        self.logprior = np.where(allowed, np.log(w), -np.inf)


base._Data.__init__ = _init
fit_predict = base.fit_predict
