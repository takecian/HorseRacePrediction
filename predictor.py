# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import argparse
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList, FunctionSet
from chainer import training
from chainer.datasets import tuple_dataset
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import pickle as P
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units\
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_units),  # n_units -> n_units
            l4=L.Linear(None, n_units),  # n_units -> n_units
            l5=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)

class DataLoader:
  def __init__ (self) :
    self.model = None

    self.train_data = np.array([], dtype=np.float32)
    self.train_data_answer = np.array([], dtype=np.int32)
    # self.train_data_answer = np.array([], dtype=np.int32).reshape(0, 18)

    # テスト対象
    self.test_data = np.array([], dtype=np.float32)
    self.test_data_answer = np.array([], dtype=np.int32)
    # self.test_data_answer = np.array([], dtype=np.int32).reshape(0, 18)

  def load(self):
    with open('train_data.pickle', 'rb') as f:
      self.train_data = P.load(f)
    print("train_data count = {}".format(len(self.train_data)))

    with open('train_data_answer.pickle', 'rb') as f:
      self.train_data_answer = P.load(f)
    print("train_data_answer count = {}".format(len(self.train_data_answer)))

    with open('test_data.pickle', 'rb') as f:
      self.test_data = P.load(f)
    print("test_data count = {}".format(len(self.test_data)))

    with open('test_data_answer.pickle', 'rb') as f:
      self.test_data_answer = P.load(f)
    print("test_data_answer count = {}".format(len(self.test_data_answer)))
  
    print("loader.train_data = {}, shape = {}".format(self.train_data.dtype, self.train_data.shape))
    print("loader.train_data_answer = {}, shape = {}".format(self.train_data_answer.dtype, self.train_data_answer.shape))
    print("loader.test_data = {}, shape = {}".format(self.test_data.dtype, self.test_data.shape))
    print("loader.test_data_answer = {}, shape = {}".format(self.test_data_answer.dtype, self.test_data_answer.shape))

def main():
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
  parser.add_argument('--epoch', '-e', type=int, default=40, help='Number of sweeps over the dataset to train')
  parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
  parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
  parser.add_argument('--unit', '-u', type=int, default=600, help='Number of units')
  args = parser.parse_args()

  print('GPU: {}'.format(args.gpu))
  print('# unit: {}'.format(args.unit))
  print('# Minibatch-size: {}'.format(args.batchsize))
  print('# epoch: {}'.format(args.epoch))
  print('')

  loader = DataLoader()
  loader.load()

  model = L.Classifier(MLP(args.unit, 18))
  if args.gpu >= 0:
      chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
      model.to_gpu()  # Copy the model to the GPU

  # Setup an optimizer
  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)

  train = tuple_dataset.TupleDataset(loader.train_data, loader.train_data_answer)
  test = tuple_dataset.TupleDataset(loader.test_data, loader.test_data_answer)

  # return

  train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
  test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

  # Set up a trainer
  updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

  # Evaluate the model with the test dataset for each epoch
  trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

  # Dump a computational graph from 'loss' variable at the first iteration
  # The "main" refers to the target link of the "main" optimizer.
  trainer.extend(extensions.dump_graph('main/loss'))

  # Take a snapshot at each epoch
  trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

  # Write a log of evaluation statistics for each epoch
  trainer.extend(extensions.LogReport())

  # Print selected entries of the log to stdout
  # Here "main" refers to the target link of the "main" optimizer again, and
  # "validation" refers to the default name of the Evaluator extension.
  # Entries other than 'epoch' are reported by the Classifier link, called by
  # either the updater or the evaluator.
  trainer.extend(extensions.PrintReport(
      ['epoch', 'main/loss', 'validation/main/loss',
       'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

  # Print a progress bar to stdout
  trainer.extend(extensions.ProgressBar())

  if args.resume:
      # Resume from a snapshot
      chainer.serializers.load_npz(args.resume, trainer)

  # Run the training
  trainer.run()

if __name__ == '__main__':
    main()
