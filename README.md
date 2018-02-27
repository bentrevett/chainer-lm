# chainer-lm

Example showing how you can use the Chainer PTB LM on other datasets.

Example adapted from: https://github.com/chainer/chainer/blob/v3/examples/ptb/train_ptb.py

Here we use it on the WikiText dataset: https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

Usage:

`python train_lm.py --gpu=0` for GPU

`python train_lm.py` for CPU 

By default it works on a word-based model, to work on characters, pass the `--chars` flag, i.e.:

`python train_lm.py --chars` for character level on CPU
