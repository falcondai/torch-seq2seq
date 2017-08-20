from __future__ import absolute_import, division, print_function
from six.moves import xrange

from util import logging, tt
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

import numpy as np
import torch
import torchvision as tv
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import tqdm
# Tensorflow imports for writing summaries
try:
    from tensorflow import summary, Summary
    from tensorflow.contrib.util import make_tensor_proto
except:
    logger.warn('tensorflow cannot be imported')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.linear = nn.Linear(20 * 24 * 24, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = x.view(-1, 20 * 24 * 24)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size for computations.')
    parser.add_argument('-n', '--n-epochs', type=int, default=6, help='Epochs to train.')
    parser.add_argument('-l', '--log-dir', default='/tmp', help='Log directory path.')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const', default=logging.INFO, const=logging.DEBUG, help='Display more log messages.')

    args = parser.parse_args()

    # Set the logging level
    logger.setLevel(args.log_level)

    # Define the model
    net = Net().cuda()
    # net = nn.DataParallel(net, [0, 1])

    logger.info(net)
    logger.info('model parameters:')
    n_params = 0
    for param in net.parameters():
        n_params += param.nelement()
        logger.info((param.size(), param.nelement()))
    logger.info('# parameters: %i', n_params)

    train_pairs = tv.datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=tv.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True)
    logger.info('# train samples: %i', len(train_pairs))

    test_pairs = tv.datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=tv.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=args.batch_size, shuffle=False)
    logger.info('# test samples: %i', len(test_pairs))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Summary
    writer = summary.FileWriter(args.log_dir, flush_secs=10)

    # Train for a few epochs
    step = 0
    for i in xrange(args.n_epochs):
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # Build a mini-batch
            imgs, labels = batch
            imgs = Variable(imgs.cuda())
            # labels.random_(10)
            labels = Variable(labels.cuda())

            preds = net(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()
            summary_proto = Summary(value=[Summary.Value(tag='train/loss', simple_value=loss.data.cpu().numpy()[0])])
            writer.add_summary(summary_proto, global_step=step)

            step += 1
            optimizer.step()

        # Evaluate on test
        n_correct = 0.
        for batch in tqdm.tqdm(test_loader):
            imgs, labels = batch
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.cuda())

            _, preds = net(imgs).max(1)
            n_correct += preds.eq(labels).float().sum().data.cpu().numpy()[0]
        accuracy = n_correct / len(test_pairs)
        logger.info('test accuracy %g', accuracy)
        writer.add_summary(Summary(value=[Summary.Value(tag='test/accuracy', simple_value=accuracy)]), global_step=i)
    writer.close()
