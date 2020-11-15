import argparse
import os
import shutil
import random
import collections
import random
from random import Random
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchaudio import transforms
from typing_extensions import Final
from typing import Optional, Union, List, Dict, Any, Tuple


def pair_speaker_samples(dataset: List[str], randomize: bool,
                         n_speakers: Optional[int] = None) -> List[Tuple[int, int]]:
    sample_pairs: List[Tuple[int, int]] = []

    # LibriSpeech doesn't have that many entires so we don't need to be pedantic about efficiency here.
    speakers_set = set([])
    speaker_sample_indices: Dict[str, List[int]] = collections.defaultdict(list)
    for i, fileid in enumerate(dataset):
        (speaker_id, _, _) = fileid.split("-")
        speaker_sample_indices[speaker_id].append(i)
        speakers_set.add(speaker_id)
    speakers = list(speakers_set)

    if n_speakers:
        speakers = random.sample(speakers, n_speakers)

    # Same class pairs.
    for samples in speaker_sample_indices.values():
        indices = list(range(0, len(samples)))

        # Randomize pair selection.
        if randomize:
            random.shuffle(indices)

        for i in range(0, len(indices)-1, 2):
            sample1_idx = indices[i]
            sample2_idx = indices[i+1]
            sample_pairs.append((samples[sample1_idx], samples[sample2_idx]))

    # Randomize pairings of different speakers.
    if randomize:
        random.shuffle(speakers)

    unused_samples = 0
    unused = []
    # Diff class pairs.
    for i in range(0, len(speakers)-1, 2):
        speaker1_samples = speaker_sample_indices[speakers[i]]
        speaker2_samples = speaker_sample_indices[speakers[i+1]]

        # Since different speakers have different amount of samples this is going to leave a few unused.
        if len(speaker1_samples) != len(speaker2_samples):
            unused_samples += abs(len(speaker1_samples) - len(speaker2_samples))

        # Randomize paired samples.
        if randomize:
            random.shuffle(speaker1_samples)
            random.shuffle(speaker2_samples)

        s1_len = len(speaker1_samples)
        s2_len = len(speaker2_samples)
        if s1_len > s2_len:
            unused.extend(speaker1_samples[s2_len:s1_len])
        elif s1_len < s2_len:
            unused.extend(speaker2_samples[s1_len:s2_len])

        sample_pairs.extend(zip(speaker1_samples, speaker2_samples))

    # Add all the pairs that were left over.
    assert unused_samples == len(unused)
    sample_pairs.extend(zip(unused[::2], unused[1::2]))

    print("Collected {} sample pairs.".format(len(sample_pairs)))

    return sample_pairs


# Pad shorter and clip longer waveforms and drop redundant channel dimension,
# since we work with only one channel.
def process_waveform(waveform, length: Optional[int] = None):
    wf_len = waveform.size(1)
    assert wf_len > 0

    if length is None:
        length = wf_len

    # Pad if too small, else pick random starting point for the slice.
    if wf_len < length:
        waveform = F.pad(waveform, (0, length - wf_len))
        wf_len = waveform.size(1)
        start = 0
    else:
        start = np.int64(random.random() * (wf_len - length))
    assert start + length <= wf_len

    # Drop channel dimension and clip to length.
    waveform = waveform.squeeze()[start:start + length]
    assert waveform.size(0) == length
    return waveform


# FIXME: Dataset classes should be refactored to have no state/rng.
#        Randomization should happen in the DataLoader/Sampler.
class PairDataset(Dataset):
    def __init__(self, dataset: dset.LIBRISPEECH,
                 n_speakers: Optional[int] = None,
                 max_sample_length: Optional[int] = None,
                 randomize: bool = True):
        super().__init__()
        self.dataset: Final = dataset
        self.samples: Final = pair_speaker_samples(dataset._walker, randomize=randomize, n_speakers=n_speakers)
        self._max_length: Final = max_sample_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (i, j) = self.samples[index]
        (waveform1, _, _, speaker1, _, _) = self.dataset.__getitem__(i)
        (waveform2, _, _, speaker2, _, _) = self.dataset.__getitem__(j)

        waveform1 = process_waveform(waveform1, self._max_length)
        waveform2 = process_waveform(waveform2, self._max_length)
        assert self._max_length is None or waveform1.size(0) == waveform2.size(0) == self._max_length

        label = 1.0 if speaker1 == speaker2 else 0.0
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (waveform1, waveform2, y)


# Returns 3 waveforms of which first 2 are from the same speaker and
# the 3rd waveform is from a different speaker.
class TripletDataset(Dataset):
    def __init__(self, dataset: dset.LIBRISPEECH,
                 n_speakers: Optional[int] = None,
                 max_sample_length: Optional[int] = None,
                 max_samples: int = 0):
        super().__init__()
        self.dataset: Final = dataset
        self.max_samples: Final = max_samples
        self._max_length: Final = max_sample_length

        # LibriSpeech doesn't have that many entires so we don't need to be pedantic about efficiency here.
        speakers_set = set([])
        self._speaker_sample_indices: Dict[str, List[int]] = collections.defaultdict(list)
        for i, fileid in enumerate(dataset._walker):
            (speaker_id, _, _) = fileid.split("-")
            self._speaker_sample_indices[speaker_id].append(i)
            speakers_set.add(speaker_id)

        self._speakers = list(speakers_set)
        if n_speakers:
            self._speakers = random.sample(self._speakers, n_speakers)

        self._length = sum([len(x) for x in self._speaker_sample_indices.values()])
        print("Created a TripletDataset of {} samples, of which {} will be used."
              .format(self._length, max(self._length, self.max_samples)))

    def __len__(self):
        return max(self._length, self.max_samples)

    def __getitem__(self, index):
        speaker1_id, speaker2_id = random.sample(self._speakers, 2)
        i, j = random.sample(self._speaker_sample_indices[speaker1_id], 2)
        k = random.choice(self._speaker_sample_indices[speaker2_id])

        (waveform1, _, _, speaker1a, _, _) = self.dataset.__getitem__(i)
        (waveform2, _, _, speaker1b, _, _) = self.dataset.__getitem__(j)
        (waveform3, _, _, speaker2, _, _) = self.dataset.__getitem__(k)
        assert speaker1a == speaker1b != speaker2

        waveform1 = process_waveform(waveform1, self._max_length)
        waveform2 = process_waveform(waveform2, self._max_length)
        waveform3 = process_waveform(waveform3, self._max_length)
        assert (self._max_length is None
                or waveform1.size(0) == waveform2.size(0) == waveform3.size(0) == self._max_length)

        return (waveform1, waveform2, waveform3)
