import numpy as np
import torch
import torch.nn.functional as F
from morse.generators import volume_sinusoid_variation


rng = np.random.default_rng()

def sample_uniform_value(min_bound, max_bound):
    return torch.rand(1).item() * (max_bound - min_bound) + min_bound

def rotation_transform(tensor: torch.Tensor):
    threshold = rng.integers(0, tensor.shape[0])
    result = torch.concat([tensor[threshold:], tensor[:threshold]], dim=0)
    return result

def normalize_mel_spec(mel_spec: torch.Tensor):
    return mel_spec / torch.max(mel_spec)

def make_runtime_rotation_transform(p=0.7):
    def tr(mel: torch.Tensor):
        if sample_uniform_value(0, 1) > p:
            return mel
        threshold = rng.integers(0, mel.shape[0])
        result = torch.concat([mel[threshold:], mel[:threshold]], dim=0)
        return result
    return tr

def make_volume_signal_transform(min_res = 0.2, max_freq = 1):
    def tr(signal):
        min_residual = rng.uniform(min_res, 1.0)
        phase = rng.uniform(0, 1) * np.pi * 2
        var_freq = rng.uniform(0.1, max_freq)
        return volume_sinusoid_variation(signal, duration=8, variation_frequency=var_freq, phase = phase, min_residual=min_residual)
    return tr

def make_noise_signal_transform(min_volume=0, max_volume=1.2, p=0.5):
    def tr(signal):
        if sample_uniform_value(0, 1) > p:
            return signal
        volume = rng.uniform(min_volume, max_volume)
        noise = torch.randn_like(signal) * volume
        return signal + noise
    return tr

def make_runtime_mel_bounded_noise_transform(std_frac_bounds=(0.015, 0.1), volume_bounds=(0, 0.5), p=0.8):
    def tr(mel: torch.Tensor):
        if sample_uniform_value(0, 1) > p:
            return mel
        std = sample_uniform_value(*std_frac_bounds)
        volume = sample_uniform_value(*volume_bounds)
        mean_frac_deviation = sample_uniform_value(-std, std)
        signal_pos_frac = torch.argmax(torch.mean(mel, dim=1)) / mel.shape[0]
        mean_frac = signal_pos_frac + mean_frac_deviation
        mask = torch.exp(-torch.square((mean_frac - torch.linspace(0, 1, steps=mel.shape[0])[:, None]) / std))
        raw_noise = torch.rand_like(mel)
        noise = raw_noise * mask * volume
        return normalize_mel_spec(noise + mel)
    return tr

def make_compose_transform(transforms):
    def tr(signal):
        for transform in transforms:
            signal = transform(signal)
        return signal
    return tr

def make_mel_pad_augmentation(pad=(0, 11)):
    def tr(mel):
        return F.pad(mel, pad=pad)
    return tr
