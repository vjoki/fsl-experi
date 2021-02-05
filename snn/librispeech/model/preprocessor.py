import platform
from typing import Optional
from typing_extensions import Final
import torch
import torch.nn as nn
import torchaudio
from resnet.utils import PreEmphasis

from snn.librispeech.dataset.util import compose_torch_augmentations

if platform.system().lower().startswith('win'):
    torchaudio.set_audio_backend("soundfile")
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
else:
    torchaudio.set_audio_backend("sox_io")


class PreProcessor(nn.Module):
    def __init__(self, signal_transform: str,
                 sample_rate: int, n_fft: int, n_mels: int,
                 specaugment: bool = False,
                 torch_augment: bool = False, rir_paths: str = '', **kwargs):
        super().__init__()
        self.instancenorm: Final[nn.Module] = nn.InstanceNorm1d(n_mels)
        # 1xTIME*SAMPLERATE -> 1xN_MELSxTIME?
        self.specaugment: Final = specaugment
        self.signal_transform: Final = signal_transform

        transform_fn: nn.Module
        if signal_transform == 'melspectrogram':
            transform_fn = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                     win_length=400, hop_length=160,
                                                     window_fn=torch.hamming_window, n_mels=n_mels)
            )
        elif signal_transform == 'spectrogram':
            transform_fn = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=400, hop_length=160,
                                                  window_fn=torch.hamming_window)
            )
        elif signal_transform == 'mfcc':
            transform_fn = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mels, log_mels=True)
            )

        if torch_augment and rir_paths:
            transform_fn = torch.nn.Sequential(
                compose_torch_augmentations(rir_paths),
                transform_fn
            )
        self.signal_transform_fn: Final[nn.Module] = transform_fn

        # Partial SpecAugment, if toggled.
        self.augment_spectrogram: Optional[nn.Module] = None
        if self.specaugment and signal_transform != 'mfcc':
            F = 0.20
            T = 0.10
            self.augment_spectrogram = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1))),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1)))
            )

    def forward(self, x: torch.Tensor, augmentable: bool = False) -> torch.Tensor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.signal_transform_fn(x)+1e-6
                if self.signal_transform != 'mfcc':
                    x = x.log()
                if augmentable and self.augment_spectrogram:
                    x = self.augment_spectrogram(x)
                x = self.instancenorm(x).unsqueeze(1)
        return x
