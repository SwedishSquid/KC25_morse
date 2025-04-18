import numpy as np
from tqdm import tqdm
import torch
from morse.text_helpers import encode_to_morse
import torch.nn.functional as F
import torchaudio


all_symbols = ' #0123456789АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
rng = np.random.default_rng()


def sample_random_text(length_bounds=(6, 12)):
    length = rng.integers(length_bounds[0], length_bounds[1] + 1)
    message_arr = []
    for i in range(length):
        while True:
            idx = rng.integers(0, len(all_symbols))
            symbol = all_symbols[idx]
            if symbol == ' ' and len(message_arr) != 0 and message_arr[-1] == ' ':
                continue
            message_arr.append(symbol)
            break
    return ''.join(message_arr)


def volume_sinusoid_variation(signal, duration, variation_frequency: float, phase, min_residual):
    t = torch.linspace(0, duration, signal.shape[0])
    volume_multiplier = 1 * min_residual + (1 - min_residual) * (1 + torch.sin(2 * torch.pi * variation_frequency * t + phase)) / 2
    return signal * volume_multiplier


def mean_normalize(audio_tensor):
    mean = torch.mean(audio_tensor)
    return audio_tensor - mean

def peak_normalize(audio_tensor):
    max_val = torch.max(torch.abs(audio_tensor))
    return audio_tensor / max_val


def normalize_audio(audio_tensor):
    return peak_normalize(mean_normalize(audio_tensor))


class MorseGenerator:
    def __init__(self, carrier_freq_range=(600, 1200), dot_duration_range=(0.04, 0.06), var_freq_bounds = (0.1, 0.5), lowest_min_residual=0.65, duration=8, sr = 8000):
        self.carrier_freq_range=carrier_freq_range
        self.dot_duration_range=dot_duration_range
        self.var_freq_bounds = var_freq_bounds
        self.lowest_min_residual = lowest_min_residual
        self.duration=duration
        self.sample_rate = sr
        pass

    def _dash_duration(self, dot_duration):
        return 3 * dot_duration
    
    def _symbol_space_duration(self, dot_duration):
        return dot_duration
    
    def _char_space_duration(self, dot_duration):
        return 3 * dot_duration
    
    def _word_space_duration(self, dot_duration):
        return 7 * dot_duration

    def _generate_signal(self, duration, freq):
        t = torch.linspace(0, duration, int(duration * self.sample_rate), dtype=torch.float32)
        return torch.sin(2 * torch.pi * freq * t)
    
    def _generate_silence(self, duration):
        return torch.zeros(int(duration * self.sample_rate), dtype=torch.float32)

    def _generate_morse_element(self, symbol, freq, dot_duration):
        if symbol == ' ':
            return self._generate_silence(self._char_space_duration(dot_duration))
        if symbol == '/':
            return self._generate_silence(self._word_space_duration(dot_duration))
        if symbol == '-':
            return self._generate_signal(self._dash_duration(dot_duration), freq)
        if symbol == '.':
            return self._generate_signal(dot_duration, freq)
        raise ValueError(f'unknown symbol {symbol}')

    def _generate_pure_sample(self, morse: str):
        low_freq, high_freq = self.carrier_freq_range
        low_dd, high_dd = self.dot_duration_range
        carrier_freq = torch.rand(1).item() * (high_freq - low_freq) + low_freq
        dot_duration = torch.rand(1).item() * (high_dd - low_dd) + low_dd
        audio_elements = []
        for i, symbol in enumerate(morse):
            if i > 0 and (symbol != ' ') and (morse[i-1] != ' '):
                audio_elements.append(self._generate_silence(self._symbol_space_duration(dot_duration)))
            audio_elements.append(self._generate_morse_element(symbol, carrier_freq, dot_duration))
        
        signal_only = torch.cat(audio_elements, dim=0)
        desired_length = int(self.duration * self.sample_rate)
        delta_length = desired_length - signal_only.shape[0]
        if delta_length < 0:
            return None # to regenerate if needed
        left_pad = int(torch.rand(1).item() * delta_length)
        right_pad = delta_length - left_pad
        padded_signal = F.pad(signal_only, (left_pad, right_pad), value=0)
        assert padded_signal.shape[0] == desired_length
        return padded_signal, carrier_freq
    
    def _generate_list_of_pure_samples(self, size):
        signals = []
        messages = []
        frequencies = []
        for i in tqdm(range(size)):
            out = None
            message = None
            while out == None:
                message = sample_random_text()
                morse = encode_to_morse(message)
                out = self._generate_pure_sample(morse)
            sample, carrier_freq = out
            signals.append(sample)
            messages.append(message)
            frequencies.append(carrier_freq)
        return signals, messages, frequencies
    
    def generate(self, size):
        print('making pure signals')
        signals, messages, frequencies = self._generate_list_of_pure_samples(size)
        print('adding noise')
        
        result_signals = []
        for signal, freq in zip(tqdm(signals), frequencies):
            assert signal.ndim == 1
            phase = torch.rand(1).item() * torch.pi * 2
            var_freq = torch.rand(1).item() * (self.var_freq_bounds[1] - self.var_freq_bounds[0]) + self.var_freq_bounds[0]
            min_residual = torch.rand(1).item() * self.lowest_min_residual
            varied_signal = volume_sinusoid_variation(signal, self.duration, var_freq, phase, min_residual=min_residual)
            volume = torch.rand(1).item() * (1.1 - 0.9) + 0.9

            volumed_signal = varied_signal * volume

            pure_noise = torch.randn(signal.shape[0])
            low_freq, high_freq = self.carrier_freq_range
            central_freq = (torch.rand(1).item() * (high_freq - low_freq) + low_freq) * 0.5 + freq * 0.5
            Q = torch.rand(1).item() * (4 - 1) + 1
            filtered_noise = torchaudio.functional.bandpass_biquad(pure_noise, sample_rate=self.sample_rate, 
                                                       central_freq=central_freq, Q=3)
            noised_signal = volumed_signal + filtered_noise

            normalized_signal = normalize_audio(noised_signal)
            # normalized_signal = noised_signal
            result_signals.append(normalized_signal)
        return result_signals, messages
