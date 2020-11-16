# Few-shot learning experiments
Dumping ground for miscellaneous ML experiments with focus on FSL.

## Dependencies
- Python 3.8
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [PyTorch](https://pytorch.org/)
- [conda](https://github.com/conda/conda)

Using conda to manage dependencies. Detailed list of dependencies in
[`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt).

## Experiments
Divided in modules by method, which are further divided into submodules by
dataset.

### Siamese network (SNN)
- [`snn/omniglot/`](snn/omniglot/): Convolutional SNN for one-shot learning on
  [Omniglot](https://github.com/brendenlake/omniglot) dataset<sup>[[1](#references)]</sup>.
  - Heavily based on reimplementations of the paper at
    [https://github.com/kevinzakka/one-shot-siamese](https://github.com/kevinzakka/one-shot-siamese) and
    [https://github.com/fangpin/siamese-pytorch](https://github.com/kevinzakka/one-shot-siamese).
  - Using the [learning rate
    finder](https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html)
    from PyTorch Lightning.
  - AdamW optimizer<sup>[[2](#references)]</sup>, with 1cycle learning rate policy<sup>[[3](#references), [4](#references)]</sup>.
- [`snn/librispeech/`](snn/librispeech/): Siamese capsule network using Thin-ResNet34
  for one-shot learning on [LibriSpeech](http://www.openslr.org/12/) dataset.
  - Experimenting based on ideas from paper by Hajavi et al.
    <sup>[[5](#references)]</sup>.
  - Thin-ResNet34 implementation copied from
    [https://github.com/clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).
  - CapsNet implementation copied from
    [https://github.com/adambielski/CapsNet-pytorch](https://github.com/adambielski/CapsNet-pytorch).
  - Using the [learning rate
    finder](https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html)
    from PyTorch Lightning.
  - Optional spectogram frequency and time masking as per SpecAugment<sup>[[6](#references)]</sup>.
  - AdamW optimizer<sup>[[2](#references)]</sup>, with 1cycle learning rate
    policy<sup>[[3](#references), [4](#references)]</sup>.

## Usage
```shell
python -m <model>.<dataset>.train --help
```

**Example**: train model `snn/omniglot/` using 1 GPU:
```shell
python -O -m snn.omniglot.train --gpus 1 --num_workers 4 --batch_size 128 --max_epochs 50
```

## References
1. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese
   neural networks for one-shot image recognition." In *ICML deep learning
   workshop*, vol. 2. 2015.
2. Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay regularization."
   *arXiv preprint arXiv:1711.05101* (2017). [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101).
3. Smith, Leslie N., and Nicholay Topin. "Super-convergence: Very fast training
   of neural networks using large learning rates." In *Artificial Intelligence and
   Machine Learning for Multi-Domain Operations Applications.* Vol. 11006.
   International Society for Optics and Photonics, 2019. [https://arxiv.org/abs/1708.07120](https://arxiv.org/abs/1708.07120).
4. https://sgugger.github.io/the-1cycle-policy.html
5. Hajavi, Amirhossein, and Ali Etemad. "Siamese Capsule Network for End-to-End
   Speaker Recognition In The Wild." *arXiv preprint arXiv:2009.13480* (2020).
   [https://arxiv.org/abs/2009.13480](https://arxiv.org/abs/2009.13480).
6. Park, Daniel S., Yu Zhang, Chung-Cheng Chiu, Youzheng Chen, Bo Li, William
   Chan, Quoc V. Le, and Yonghui Wu. "Specaugment on large scale datasets." In
   *ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and
   Signal Processing (ICASSP)*, pp. 6879-6883. IEEE, 2020.
   [https://arxiv.org/abs/1904.08779](https://arxiv.org/abs/1904.08779).
