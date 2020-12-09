import os
import random
import collections
from typing import Optional, List, Dict, Tuple
from typing_extensions import Final
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.datasets as dset
from torch.utils.data.dataset import Dataset


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

    n_same_class_pairs = len(sample_pairs)

    # Randomize pairings of different speakers.
    if randomize:
        random.shuffle(speakers)

    # Since different speakers have different amount of samples this is going to leave a few unused.
    unused = []
    # Diff class pairs.
    for i in range(0, len(speakers)-1, 2):
        speaker1_samples = speaker_sample_indices[speakers[i]]
        speaker2_samples = speaker_sample_indices[speakers[i+1]]

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

    n_diff_class_pairs = len(sample_pairs) - n_same_class_pairs

    # Add all the pairs that were left over.
    sample_pairs.extend(zip(unused[::2], unused[1::2]))

    print("Collected {} sample pairs. {} speakers. {} same speaker pairs. {} different speaker pairs. {} mixed pairs."
          .format(len(sample_pairs),
                  n_speakers if n_speakers else len(speakers),
                  n_same_class_pairs,
                  n_diff_class_pairs,
                  len(unused)))

    return sample_pairs


# Pad shorter and clip longer waveforms and drop redundant channel dimension,
# since we work with only one channel.
def process_waveform(waveform, max_frames_per_sample: Optional[int] = None):
    if waveform.dim() >= 2:
        assert waveform.size(0) == 1
        waveform = waveform.squeeze()

    num_frames = waveform.size(0)
    assert num_frames > 0

    if max_frames_per_sample is None:
        max_frames_per_sample = num_frames

    # Pad if too small, else pick random starting point for the slice.
    if num_frames < max_frames_per_sample:
        waveform = F.pad(waveform, (0, max_frames_per_sample - num_frames))  # type: ignore
        num_frames = waveform.size(0)
        start = 0
    else:
        start = int(random.random() * (num_frames - max_frames_per_sample))
    assert start + max_frames_per_sample <= num_frames

    # Drop channel dimension and clip to length.
    waveform = waveform[start:start + max_frames_per_sample]
    assert waveform.size(0) == max_frames_per_sample
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
        self.pairs: Final = pair_speaker_samples(dataset._walker, randomize=randomize, n_speakers=n_speakers)
        self._max_length: Final = max_sample_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (i, j) = self.pairs[index]
        (waveform1, sample_rate, _, speaker1, _, _) = self.dataset.__getitem__(i)
        (waveform2, _, _, speaker2, _, _) = self.dataset.__getitem__(j)

        max_frames = self._max_length * sample_rate if self._max_length else None

        waveform1 = process_waveform(waveform1, max_frames_per_sample=max_frames)
        waveform2 = process_waveform(waveform2, max_frames_per_sample=max_frames)
        assert self._max_length is None or waveform1.size(0) == waveform2.size(0) == max_frames

        label = 1.0 if speaker1 == speaker2 else 0.0
        y = torch.as_tensor([label])
        return (waveform1, waveform2, y)


class PairDatasetFromList(Dataset):
    def __init__(self,
                 list_file,
                 data_path,
                 max_sample_length = None,
                 rir_path: str = './data/RIRS_NOISES/',
                 augment: bool = False):
        super().__init__()
        self.data_path = data_path
        self._max_length = max_sample_length
        self._augment: Final = augment
        self._transform: Final = Compose([
            AddGaussianSNR(min_SNR=0.2, max_SNR=0.5, p=0.5),
            AddImpulseResponse(os.path.join(rir_path, 'real_rirs_isotropic_noises'), p=0.5),
            AddShortNoises(os.path.join(rir_path, 'pointsource_noises'), p=0.5)
        ])
        with open(list_file) as f:
            self.pairs = [line.rstrip().split(" ") for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (label, a, b) = self.pairs[index]

        waveform1, sample_rate = torchaudio.load_wav(os.path.join(self.data_path, a))
        waveform2, _ = torchaudio.load_wav(os.path.join(self.data_path, b))
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

        (waveform1, sample_rate, _, speaker1a, _, _) = self.dataset.__getitem__(i)
        (waveform2, _, _, speaker1b, _, _) = self.dataset.__getitem__(j)
        (waveform3, _, _, speaker2, _, _) = self.dataset.__getitem__(k)
        assert speaker1a == speaker1b != speaker2

        max_frames = self._max_length * sample_rate if self._max_length else None

        waveform1 = process_waveform(waveform1, max_frames_per_sample=max_frames)
        waveform2 = process_waveform(waveform2, max_frames_per_sample=max_frames)
        waveform3 = process_waveform(waveform3, max_frames_per_sample=max_frames)
        assert (max_frames is None
                or waveform1.size(0) == waveform2.size(0) == waveform3.size(0) == max_frames)

        return (waveform1, waveform2, waveform3)
