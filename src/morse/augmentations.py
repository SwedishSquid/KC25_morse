import numpy as np
import torch
from morse.generators import volume_sinusoid_variation


rng = np.random.default_rng()

def rotation_transform(tensor: torch.Tensor):
    threshold = rng.integers(0, tensor.shape[0])
    result = torch.concat([tensor[threshold:], tensor[:threshold]], dim=0)
    return result


def make_volume_signal_transform(min_res = 0.2, max_freq = 1):
    def tr(signal):
        min_residual = rng.uniform(min_res, 1.0)
        phase = rng.uniform(0, 1) * np.pi * 2
        var_freq = rng.uniform(0.1, max_freq)
        return volume_sinusoid_variation(signal, duration=8, variation_frequency=var_freq, phase = phase, min_residual=min_residual)
    return tr

def make_noise_signal_transform(min_volume=0, max_volume=1.2):
    def tr(signal):
        volume = rng.uniform(min_volume, max_volume)
        noise = torch.randn_like(signal) * volume
        return signal + noise
    return tr

def make_compose_signal_transform(transforms):
    def tr(signal):
        for transform in transforms:
            signal = transform(signal)
        return signal
    return tr
