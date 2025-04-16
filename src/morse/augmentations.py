import numpy as np
import torch


rng = np.random.default_rng()

def rotation_transform(tensor: torch.Tensor):
    threshold = rng.integers(0, tensor.shape[0])
    result = torch.concat([tensor[threshold:], tensor[:threshold]], dim=0)
    return result