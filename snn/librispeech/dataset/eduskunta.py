import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    walk_files,
)


def load_eduskunta_item(fileid: str,
                        path: str,
                        ext_audio: str,
                        ext_txt: str) -> Tuple[Tensor, int, str, int, int]:
    speaker_id, utterance_id, _, _ = fileid.split("_")

    file_text = fileid + ext_txt
    file_text = os.path.join(path, speaker_id, file_text)

    file_audio = fileid + ext_audio
    file_audio = os.path.join(path, speaker_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text, encoding='ISO-8859-1') as ft:
        utterance = ft.read()

    return (
        waveform,
        sample_rate,
        utterance,
        int(speaker_id),
        int(utterance_id),
    )


class EDUSKUNTA(Dataset):
    """
    Create a Dataset for LibriSpeech. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self, root: str) -> None:

        self._path = os.path.join(root, 'edus80')

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int]:
        fileid = self._walker[n]
        return load_eduskunta_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
