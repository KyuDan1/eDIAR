# Vendored from https://github.com/facebookresearch/faiss/tree/main/contrib/e_means (MIT, Meta Platforms).
# Only the CPU/NumPy path is copied, unmodified: base.py, e_means.py, e_means_spherical.py.
# Paper: Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501.
from .e_means import NumpyEmeans

__all__ = ["NumpyEmeans"]
