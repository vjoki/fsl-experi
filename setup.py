#!/usr/bin/env python
from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='fsl-experi',
      description='FSL experiments',
      author='Ville Jokinen',
      author_email='vjoki@zv.fi',
      version='0.1',
      install_requires=requirements,
      packages=['snn'],
      entry_points={
          'console_scripts': [
              'snn.train = snn.omniglot.train:main'
              'snn.infer = snn.omniglot.infer:main'
          ]
      })
