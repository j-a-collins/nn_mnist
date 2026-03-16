# MNIST from scratch with numpy

a repo for a small neural network project that trains an MNIST digit classifier from scratch using python and numpy only. no pytorch. no tensorflow. no autograd. just matrix multiplication, manual backpropagation, and minimal linear algebra. the project starts with a CPU-only implementation and later i'll add a cupy-backed GPU version.

## features

- MNIST download and parsing from the raw IDX gzip files
- fully connected neural network implemented from scratch
- manual forward pass, loss, backpropagation, and parameter updates
- mini-batch SGD training
- train and test metric tracking
- metric plotting with matplotlib
- model checkpoint saving and loading
- notebook-based inference and visualisation

## project aims

main goals are:

- understand how a neural network actually works under the hood
- do not rely on external implementations
- implement the core training loop manually
- verify that a simple multilayer perceptron can reach strong MNIST accuracy

## current model

the current model is a simple multilayer perceptron:

- input: `784`
- hidden layer: `256`
- activation: `ReLU`
- output: `10`

and in in architectural form:

```text
784 -> Dense(256) -> ReLU -> Dense(10)
```

## example
a typical run of the current CPU version reaches roughly
- train accuracy: ~97.6%
- test accuracy: ~97.2%

example output:
```
epoch=01 train_loss=0.4477 train_acc=0.8810 test_loss=0.2696 test_acc=0.9252
epoch=02 train_loss=0.2444 train_acc=0.9323 test_loss=0.2087 test_acc=0.9402
epoch=03 train_loss=0.1956 train_acc=0.9447 test_loss=0.1756 test_acc=0.9497
epoch=04 train_loss=0.1638 train_acc=0.9540 test_loss=0.1525 test_acc=0.9564
epoch=05 train_loss=0.1416 train_acc=0.9605 test_loss=0.1359 test_acc=0.9604
epoch=06 train_loss=0.1245 train_acc=0.9654 test_loss=0.1231 test_acc=0.9649
epoch=07 train_loss=0.1113 train_acc=0.9687 test_loss=0.1146 test_acc=0.9674
epoch=08 train_loss=0.1006 train_acc=0.9723 test_loss=0.1054 test_acc=0.9697
epoch=09 train_loss=0.0916 train_acc=0.9746 test_loss=0.1016 test_acc=0.9701
epoch=10 train_loss=0.0842 train_acc=0.9765 test_loss=0.0960 test_acc=0.9725
```

# requirements
- python 3.12+
- numpy
- matplotlib
- jupyter
- ipykernel
- pillow (for external image inference later)

# setup

create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

install dependencies:
```bash 
python -m pip install -U numpy matplotlib jupyter ipykernel pillow
```

# dataset

the project uses the original MNIST dataset stored as gzip-compressed IDX files. the downloader in data/mnist.py fetches:
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

these get downloaded into data/raw/.

# run it

run the MNIST loader
```bash
python -m data.mnist
```
to train the model:

```bash
python -m train
```
