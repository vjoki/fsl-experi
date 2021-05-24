# Few-shot learning experiments
Dumping ground for miscellaneous ML experiments with focus on FSL.

## Dependencies
- Python 3.8
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [PyTorch](https://pytorch.org/)
- [conda](https://github.com/conda/conda)

Using conda to manage dependencies. Detailed list of dependencies in
[`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt).

## Eduskunta dataset
[EDUSKUNTA.md](EDUSKUNTA.md) contains a guide for compiling speaker recognition
dataset consisting of Finnish speech from the The [Plenary Sessions of the
Parliament of Finland](http://urn.fi/urn:nbn:fi:lb-2017030901) dataset.

## Experiments
[`snn/librispeech/`](snn/librispeech) contains multiple speaker recognition
networks for one-shot learning on [LibriSpeech](http://www.openslr.org/12/)
dataset.

- Thin-ResNet34, fast-ResNet34, SAP and ASP implementations adapted from
  <https://github.com/clovaai/voxceleb_trainer>.
- NetVLAD and GhostVLAD pieced together from
  <https://github.com/lyakaap/NetVLAD-pytorch/>,
  <https://github.com/Nanne/pytorch-NetVlad/>,
  <https://github.com/sitzikbs/netVLAD/>.
- Using the [learning rate
  finder](https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html)
  from PyTorch Lightning.
- AdamW optimizer<sup>[[2](#references)]</sup>, with 1cycle learning rate
  policy<sup>[[3](#references), [4](#references)]</sup>.

### Useful command line options
#### General model options

- `--model snn | snn-capsnet | snn-angularproto | snn-softmaxproto`:
- `--signal_transform melspectrogram | spectrogram | mfcc`: The signal
  representation to feed to the ResNet, defaults to 'melspectrogram'.
- `--n_mels n`: Number of Mels to use for the Mel spectrogram or MFCC, defaults
  to 40.
- `--n_fft n`: The value of n_fft to use when constructing the spectrogram.

#### ResNet options:

- `--resnet_type thin | fast`: Choose either thin-ResNet34 or fast-ResNet34,
  defaults to `thin`.
- `--resnet_aggregation_type SAP | ASP | NetVLAD | GhostVLAD`: Choose the type
  of aggregation (or pooling) to use for the ResNet output, defaults to `SAP`.
- `--resnet_n_out n`: Adjust the size of the ResNet output tensor, `512` by
  default.

#### Data augmentation options

- `--augment`: Enable augmentation using
  [audiomentations](https://github.com/iver56/audiomentations).
- `--torch_augment`: Enable augmentation by
  [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations).
- `--specaugment`: Enable spectogram frequency and time masking as per
  SpecAugment<sup>[[6](#references)]</sup>.

#### Training options

- `--num_speakers n`: Number of speakers to include in the training set, defaults
  to 0 which selects all available.
- `--num_train n`: Number of random samples to take from the training set,
  defaults to the training set size but can be set higher.
- `--train_batch_size n`: The batch size to use specifically only for training.

## Networks
### `snn`
Simple end-to-end Siamese neural network using binary cross-entropy loss and
basic learning distance measure.

### `snn-angularproto`
Neural network using metric learning with angular prototypical loss
function<sup>[[7](#references)]</sup>.

Options:

- `--num_ways k`: Number of speakers (or classes) to include in each training
  step.
- `--num_shots n`: Number of samples to use per speaker.

### `snn-softmaxproto`
Like `snn-angularproto`, but using softmax prototypical
loss<sup>[[8](#references)]</sup>.

### `snn-capsnet`
Experimenting based on ideas from paper by Hajavi et al.
<sup>[[5](#references)]</sup>.

- CapsNet implementation copied from
  <https://github.com/adambielski/CapsNet-pytorch>.

### Extra
[`snn/omniglot/`](snn/omniglot/): Convolutional SNN for one-shot learning on
[Omniglot](https://github.com/brendenlake/omniglot)
dataset<sup>[[1](#references)]</sup>.
- Heavily based on reimplementations of the paper at
  <https://github.com/kevinzakka/one-shot-siamese> and
  <https://github.com/fangpin/siamese-pytorch>.
- Using the [learning rate
  finder](https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html)
  from PyTorch Lightning.
- AdamW optimizer<sup>[[2](#references)]</sup>, with 1cycle learning rate
  policy<sup>[[3](#references), [4](#references)]</sup>.

## Usage
```shell
python -m <model>.<dataset>.train --help
```

**Example**: train model `snn/omniglot/` using 1 GPU:
```shell
python -O -m snn.omniglot.train --gpus 1 --num_workers 4 --batch_size 128
--max_epochs 50
```

# References
1. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese
   neural networks for one-shot image recognition." In *ICML deep learning
   workshop*, vol. 2. 2015.
2. Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay regularization."
   *arXiv preprint arXiv:1711.05101* (2017). <https://arxiv.org/abs/1711.05101>.
3. Smith, Leslie N., and Nicholay Topin. "Super-convergence: Very fast training
   of neural networks using large learning rates." In *Artificial Intelligence
   and Machine Learning for Multi-Domain Operations Applications.* Vol. 11006.
   International Society for Optics and Photonics, 2019.
   <https://arxiv.org/abs/1708.07120>.
4. <https://sgugger.github.io/the-1cycle-policy.html>
5. Hajavi, Amirhossein, and Ali Etemad. "Siamese Capsule Network for End-to-End
   Speaker Recognition In The Wild." *arXiv preprint arXiv:2009.13480* (2020).
   <https://arxiv.org/abs/2009.13480>.
6. Park, Daniel S., Yu Zhang, Chung-Cheng Chiu, Youzheng Chen, Bo Li, William
   Chan, Quoc V. Le, and Yonghui Wu. "Specaugment on large scale datasets." In
   *ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and
   Signal Processing (ICASSP)*, pp. 6879-6883. IEEE, 2020.
   <https://arxiv.org/abs/1904.08779>.
7. Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo,
   Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee,
   Bong-Jin and Han, Icksang. "In defence of metric learning for speaker
   recognition." *Interspeech*. 2019. <https://arxiv.org/abs/2003.11982>.
8. Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son. "Clova
   baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020." *arXiv
   preprint*. 2020. <https://arxiv.org/abs/2009.14153>
