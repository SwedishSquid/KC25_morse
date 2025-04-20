import torch
from pathlib import Path
import torchaudio
from morse.generators import MorseGenerator
from tqdm import tqdm
import librosa
from morse.augmentations import normalize_mel_spec


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, features: list, labels: list, transform = lambda x: x):
        self.features = features
        self.labels = labels
        assert len(features) == len(labels)
        self.transform = transform
        pass

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.transform(self.features[index]), self.labels[index]



def load_tensors(dir_path, filenames):
    for name in filenames:
        path = Path(dir_path, name)
        yield torch.load(path, weights_only=True)



def filenames_to_torch(filenames):
    for name in filenames:
        yield Path(name).with_suffix('.pt')


sample_rate = 8000
n_mels = 64
n_fft = 512
hop_length = n_fft // 4

carrier_freq_range=(100, 2400)


signal_to_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=n_mels,
    n_fft=n_fft,
    hop_length=hop_length,
    normalized=True
)


def generate_dataset(size, signal_transform = lambda x: x, mel_spec_transform = lambda x: x,
                     runtime_transform = lambda x: x, show_pbar=True, inner_dot_duration_multiplier_range=(0.9, 1.1)):
    mel_specs = []
    messages = []
    generator = MorseGenerator(carrier_freq_range=carrier_freq_range,
                                inner_dot_duration_multiplier_range=inner_dot_duration_multiplier_range)
    iterator = generator.pure_signals_generator(size)
    if show_pbar:
        iterator = tqdm(iterator, total=size)
    for pure_signal, message in iterator:
        transformed_signal = signal_transform(pure_signal)
        mel = signal_to_mel_transform(transformed_signal)
        assert mel.ndim == 2
        mel = normalize_mel_spec(mel)
        mel = mel_spec_transform(mel)
        mel_specs.append(mel)
        messages.append(message)
    return ListDataset(mel_specs, messages, runtime_transform)


def read_dataset_from_files(audio_dir, filenames, labels, signal_transform = lambda x: x, mel_spec_transform = lambda x: x,
                             runtime_transform = lambda x: x, show_pbar=True):
    mel_specs = []
    iterator = filenames
    if show_pbar:
        iterator = tqdm(iterator)
    for name in iterator:
        signal, sr = librosa.load(Path(audio_dir, name), sr=None)
        assert sr == 8000
        signal = torch.as_tensor(signal)
        signal = signal_transform(signal)
        mel = signal_to_mel_transform(signal)
        mel = normalize_mel_spec(mel)
        mel = mel_spec_transform(mel)
        assert mel.ndim == 2
        mel_specs.append(mel)
    assert len(mel_specs) == len(labels)
    return ListDataset(mel_specs, labels, transform=runtime_transform)
