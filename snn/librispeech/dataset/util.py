import os
import random
import collections
from typing import Optional, List, Dict, Tuple
import torch.nn.functional as F
from typing_extensions import Final
from audiomentations import Compose, AddGaussianSNR, AddImpulseResponse, AddShortNoises, AddBackgroundNoise
import torch_audiomentations as ta


def get_fileid_speaker(fileid: str) -> str:
    split = fileid.split("-")
    if len(split) == 1:
        split = fileid.split("_")
    return split[0]


def pair_speaker_samples(dataset: List[str], randomize: bool,
                         n_speakers: Optional[int] = None) -> List[Tuple[int, int]]:
    sample_pairs: List[Tuple[int, int]] = []

    # LibriSpeech doesn't have that many entires so we don't need to be pedantic about efficiency here.
    speakers_set: Final = set([])
    speaker_sample_indices: Final[Dict[str, List[int]]] = collections.defaultdict(list)
    for i, fileid in enumerate(dataset):
        speaker_id = get_fileid_speaker(fileid)

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

    n_same_class_pairs: Final = len(sample_pairs)

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


def compose_augmentations(rir_path):
    impulse_path = os.path.join(rir_path, 'simulated_rirs')
    noise_path = os.path.join(rir_path, 'pointsource_noises')
    if not (os.path.exists(impulse_path) and os.path.exists(noise_path)):
        raise ValueError('Unable to augment signal, rir_path "{}" does not exist.'.format(rir_path))

    return Compose([
        AddGaussianSNR(min_SNR=0.2, max_SNR=0.5, p=0.5),
        AddImpulseResponse(impulse_path, leave_length_unchanged=True, p=0.3),
        AddBackgroundNoise(noise_path, p=0.3),
        AddShortNoises(noise_path, max_snr_in_db=80, p=0.3)
    ])


def compose_torch_augmentations(rir_path):
    impulse_path = os.path.join(rir_path, 'simulated_rirs')
    noise_path = os.path.join(rir_path, 'pointsource_noises')
    if not (os.path.exists(impulse_path) and os.path.exists(noise_path)):
        raise ValueError('Unable to augment signal, rir_path "{}" does not exist.'.format(rir_path))

    return ta.Compose(transforms=[
        ta.ApplyImpulseResponse(impulse_path, convolve_mode='same', p=0.3),
        ta.AddBackgroundNoise(noise_path, p=0.3),
        ta.Gain(min_gain_in_db=-15.0, max_gain_in_db=10.0, p=0.3)
    ])
