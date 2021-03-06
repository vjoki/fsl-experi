import collections
import random
from typing import Optional, List, Dict, Union
from typing_extensions import Final
import torch
import torchaudio.datasets as dset
from torch.utils.data.dataset import Dataset

from .eduskunta import EDUSKUNTA
from .util import process_waveform, compose_augmentations, get_fileid_speaker


class NShotKWayDataset(Dataset):
    SAMPLE_RATE: Final[int] = 16000

    def __init__(self, dataset: Union[dset.LIBRISPEECH, EDUSKUNTA],
                 n_speakers: Optional[int] = None,
                 num_shots: int = 1,
                 num_ways: int = 5,
                 rir_path: str = './data/RIRS_NOISES/',
                 augment: bool = True,
                 max_sample_length: Optional[int] = None):
        super().__init__()
        assert num_shots > 0
        assert num_ways > 0

        self.n_shots: Final = num_shots
        self.n_ways: Final = num_ways
        self.dataset: Final = dataset
        self._augment: Final = augment
        self._max_length: Final = max_sample_length
        if augment:
            self._transform: Final = compose_augmentations(rir_path)

        speaker_set = set([])
        samples: Dict[str, List[int]] = collections.defaultdict(list)
        for i, fileid in enumerate(self.dataset._walker):
            speaker_id = get_fileid_speaker(fileid)
            samples[speaker_id].append(i)
            speaker_set.add(speaker_id)
        speakers = list(speaker_set)

        if n_speakers:
            speakers = random.sample(speakers, n_speakers)
        else:
            n_speakers = len(speakers)

        self.sid_encoding: Final[Dict[str, int]] = dict(zip(speakers, range(n_speakers)))
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

        print("Created {}-shot {}-way dataset of {} tasks from {} samples using {} speakers."
              .format(self.n_shots, self.n_ways, len(self.entries), len(self.dataset), n_speakers))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        support_indices = self.entries[index]
        max_frames = self._max_length * self.SAMPLE_RATE if self._max_length else None

        support_set = torch.empty(size=(self.n_ways, self.n_shots, max_frames))
        labels = torch.empty(size=(self.n_ways, self.n_shots, 1), dtype=torch.long)
        assert support_set.device.type == labels.device.type == "cpu"

        assert len(support_indices) == self.n_ways
        for i, way in enumerate(support_indices):
            assert len(way) == self.n_shots
            for j, shot in enumerate(way):
                (wf, sample_rate, _, speaker, _, _) = self.dataset.__getitem__(shot)
                assert wf.device.type == "cpu"
                assert sample_rate == self.SAMPLE_RATE

                if self._augment:
                    assert wf.size(0) == 1
                    wf = wf.squeeze()
                    wf = torch.from_numpy(self._transform(wf.t().numpy(), sample_rate=sample_rate))

                wf = process_waveform(wf, max_frames_per_sample=max_frames)
                labels[i, j] = torch.LongTensor([self.sid_encoding[str(speaker)]])
                support_set[i, j] = wf

        return (support_set, labels)
