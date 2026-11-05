"""LU linear transforms with fixed indices that follow the model's device."""

import torch
from glasflow.nflows.transforms.lu import LULinear


class DeviceLULinear(LULinear):
    """Keep Glasflow's LU math, but transfer its indexing arrays only once.

    The parent stores lower/upper/diagonal indices as NumPy arrays. Indexing a
    CUDA tensor with those arrays repeatedly converts/transfers them during
    factor construction. Torch integer buffers instead move with ``model.to``
    and are reused by the inherited forward, inverse and weight methods.

    Indices are deterministic from ``features``, so they are non-persistent:
    checkpoint keys and trainable parameters remain identical to LULinear.
    """

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache, identity_init, eps)
        for name in ("lower_indices", "upper_indices", "diag_indices"):
            indices = torch.stack(
                [torch.as_tensor(index, dtype=torch.long) for index in getattr(self, name)]
            )
            # register_buffer requires removal of the parent's ordinary attribute.
            delattr(self, name)
            self.register_buffer(name, indices, persistent=False)
