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

## Usage
```shell
python -m <model>.<dataset>.train --help
```

**Example**: train model `snn/omniglot/` using 1 GPU:
```shell
python -m snn.omniglot.train --gpus 1 --num_workers 4 --batch_size 128 --max_epochs 50
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
