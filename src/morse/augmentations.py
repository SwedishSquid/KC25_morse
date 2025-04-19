import numpy as np
import torch
from morse.generators import volume_sinusoid_variation


rng = np.random.default_rng()

def rotation_transform(tensor: torch.Tensor):
    threshold = rng.integers(0, tensor.shape[0])
    result = torch.concat([tensor[threshold:], tensor[:threshold]], dim=0)
    return result



def volume_signal_transform(signal: torch.Tensor):
    min_residual = rng.uniform(0.3, 1.0)
    phase = rng.uniform(0, 1) * np.pi * 2
    var_freq = rng.uniform(0.1, 1)
    return volume_sinusoid_variation(signal, duration=8, variation_frequency=var_freq, phase = phase, min_residual=min_residual)
