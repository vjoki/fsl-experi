import os
from typing import Optional
from typing_extensions import Final
import torch
import torchaudio
import torchaudio.datasets as dset
from torch.utils.data.dataset import Dataset

from .util import pair_speaker_samples, process_waveform, compose_augmentations


# FIXME: Dataset classes should be refactored to have no state/rng.
#        Randomization should happen in the DataLoader/Sampler.
# 1-shot 1-way.
class PairDataset(Dataset):
    def __init__(self, dataset: dset.LIBRISPEECH,
                 n_speakers: Optional[int] = None,
                 max_sample_length: Optional[int] = None,
                 rir_path: str = './data/RIRS_NOISES/',
                 augment: bool = False,
                 randomize: bool = True):
        super().__init__()
        self.dataset: Final = dataset
        self.pairs: Final = pair_speaker_samples(dataset._walker, randomize=randomize, n_speakers=n_speakers)
        self._max_length: Final = max_sample_length
        self._augment: Final = augment
        if augment:
            self._transform: Final = compose_augmentations(rir_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (i, j) = self.pairs[index]
        (waveform1, sample_rate, _, speaker1, _, _) = self.dataset.__getitem__(i)
        (waveform2, _, _, speaker2, _, _) = self.dataset.__getitem__(j)
        assert waveform1.device.type == waveform2.device.type == "cpu"

        max_frames = self._max_length * sample_rate if self._max_length else None

        if self._augment:
            assert waveform1.size(0) == waveform2.size(0) == 1
            waveform1 = waveform1.squeeze()
            waveform1 = torch.from_numpy(self._transform(waveform1.t().numpy(), sample_rate=sample_rate))
            waveform2 = waveform2.squeeze()
            waveform2 = torch.from_numpy(self._transform(waveform2.t().numpy(), sample_rate=sample_rate))

        waveform1 = process_waveform(waveform1, max_frames_per_sample=max_frames)
        waveform2 = process_waveform(waveform2, max_frames_per_sample=max_frames)
        assert self._max_length is None or waveform1.size(0) == waveform2.size(0) == max_frames

        label = 1.0 if speaker1 == speaker2 else 0.0
        y = torch.as_tensor([label])
        assert waveform1.device.type == waveform2.device.type == y.device.type == "cpu"
        return (waveform1, waveform2, y)


class PairDatasetFromList(Dataset):
    def __init__(self,
                 list_file,
                 data_path,
                 max_sample_length: Optional[int] = None,
                 rir_path: str = './data/RIRS_NOISES/',
                 augment: bool = False):
        super().__init__()
        self.data_path = data_path
        self._max_length = max_sample_length
        self._augment: Final = augment
        if augment:
            self._transform: Final = compose_augmentations(rir_path)
        with open(list_file) as f:
            self.pairs = [line.rstrip().split(" ") for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (label, a, b) = self.pairs[index]

        waveform1, sample_rate = torchaudio.load(os.path.join(self.data_path, a))
        waveform2, _ = torchaudio.load(os.path.join(self.data_path, b))
        assert waveform1.device.type == waveform2.device.type == "cpu"

        max_frames = self._max_length * sample_rate if self._max_length else None

        if self._augment:
            assert waveform1.size(0) == waveform2.size(0) == 1
            waveform1 = waveform1.squeeze()
            waveform1 = torch.from_numpy(self._transform(waveform1.t().numpy(), sample_rate=sample_rate))
            waveform2 = waveform2.squeeze()
            waveform2 = torch.from_numpy(self._transform(waveform2.t().numpy(), sample_rate=sample_rate))

        waveform1 = process_waveform(waveform1, max_frames_per_sample=max_frames)
        waveform2 = process_waveform(waveform2, max_frames_per_sample=max_frames)
        assert self._max_length is None or waveform1.size(0) == waveform2.size(0) == max_frames

        y = torch.as_tensor([float(label)])
        assert waveform1.device.type == waveform2.device.type == y.device.type == "cpu"
        return (waveform1, waveform2, y)
