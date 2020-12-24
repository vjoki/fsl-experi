import os
import collections
from typing import Optional, List, Dict, Tuple, Set
from typing_extensions import Final
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.datasets as dset
from torch.utils.data.dataset import Dataset
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, AddImpulseResponse, AddShortNoises

from .util import process_waveform


class NShotKWayDataset(Dataset):
    def __init__(self, dataset: dset.LIBRISPEECH,
                 num_shots: int = 1,
                 num_ways: int = 5,
                 rir_path: str = './data/RIRS_NOISES/',
                 augment: bool = True,
                 random: bool = False,
                 max_sample_length: Optional[int] = None):
        super().__init__()
        assert num_shots > 0
        assert num_ways > 0

        self.n_shots: Final = num_shots
        self.n_ways: Final = num_ways
        self.dataset: Final = dataset
        self._augment: Final = augment
        self._max_length: Final = max_sample_length
        self._transform: Final = Compose([
            AddGaussianSNR(min_SNR=0.2, max_SNR=0.5, p=0.5),
            AddImpulseResponse(os.path.join(rir_path, 'real_rirs_isotropic_noises'), p=0.5),
            AddShortNoises(os.path.join(rir_path, 'pointsource_noises'), p=0.5)
        ])

        speaker_set = set([])
        samples: Dict[str, List[int]] = collections.defaultdict(list)
        for i, fileid in enumerate(self.dataset._walker):
            (speaker_id, _, _) = fileid.split("-")
            samples[speaker_id].append(i)
            speaker_set.add(speaker_id)
        speakers = list(speaker_set)

        self.entries = []

        while len(speakers) >= self.n_ways:
            support_sets: List[List[int]] = []

            c = 0
            speaker_id = speakers[c]
            while len(samples[speaker_id]) <= self.n_shots * self.n_ways:
                c += 1
                speaker_id = speakers[c]

            # Choose speakers where speaker != query speaker.
            neg_support: List[List[int]] = []
            neg_speakers = []
            remove_speakers = []
            for sid in speakers:
                if sid == speaker_id:
                    continue
                if len(samples[sid]) < self.n_shots:
                    remove_speakers.append(sid)
                    continue
                neg_speakers.append(sid)
                if len(neg_speakers) >= self.n_ways:
                    break

            # Break if we ran out of speakers to use for the support set.
            if len(neg_speakers) < self.n_ways:
                break

            # Add n-shots for the non-query speaker ways.
            for sid in neg_speakers:
                neg_support.append([samples[sid].pop() for _ in range(0, self.n_shots)])
                if len(samples[sid]) <= 0:
                    remove_speakers.append(sid)
            support_sets.extend(neg_support)

            for sid in remove_speakers:
                speakers.remove(sid)

            assert len(support_sets) == self.n_ways
            self.entries.append(support_sets)

        print("Created {}-shot {}-way dataset of {} tasks from {} samples."
              .format(self.n_shots, self.n_ways, len(self.entries), len(self.dataset)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        support_indices = self.entries[index]

        support_set = []
        labels = []

        assert len(support_indices) == self.n_ways
        for way in support_indices:
            shots = []
            shot_labels = []
            assert len(way) == self.n_shots
            for shot in way:
                (wf, sample_rate, _, speaker, _, _) = self.dataset.__getitem__(shot)
                max_frames = self._max_length * sample_rate if self._max_length else None

                if self._augment:
                    assert wf.size(0) == 1
                    wf = wf.squeeze()
                    wf = torch.from_numpy(self._transform(wf.t().numpy(), sample_rate=sample_rate))
                    if torch.isnan(wf).any() or torch.isinf(wf).any():
                        print('invalid input detected at augment', wf)

                wf = process_waveform(wf, max_frames_per_sample=max_frames)
                shots.append(wf)
                shot_labels.append(torch.LongTensor(speaker))
            support_set.append(shots)
            labels.append(shot_labels)

        return (support_set, labels)
