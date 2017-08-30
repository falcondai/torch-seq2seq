#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
from six.moves import xrange

from util import logging, make_checkpoint, global_norm
logger = logging.getLogger('train')

import os
import numpy as np
import tqdm

# Torch imports
import torch
import torchvision as tv
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from registry import problem_registry
import problems


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
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--problem', required=True, help='Problem to solve.')
    # parser.add_argument('-m', '--model', required=True, help='Model to use.')

    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size for computations.')
    parser.add_argument('-n', '--n-epochs', type=int, default=6, help='Epochs to train.')

    parser.add_argument('-s', '--seed', type=int, default=None, help='Manual random seed.')
    parser.add_argument('-g', '--gpus', nargs='+', default=[], type=int, help='GPUs to be used in computation.')
    parser.add_argument('-l', '--log-dir', default='/tmp', help='Log directory path.')
    parser.add_argument('-r', '--resume', help='Resume training from a saved checkpoint.')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const', default=logging.INFO, const=logging.DEBUG, help='Display more log messages.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Write Summary protobuf for TensorBoard visualizations. (Requires TensorFlow)')

    args, extra_args = parser.parse_known_args()

    # Set the logging level
    logger.setLevel(args.log_level)
    # Debug information
    logger.debug('torch version %s', torch.__version__)

    # Tensorflow imports for writing summaries
    try:
        from tensorflow import summary, Summary
        from tensorflow.contrib.util import make_tensor_proto
        logger.debug('imported TensorFlow.')
    except ImportError:
        logger.warn('TensorFlow cannot be imported. TensorBoard summaries will not be generated. Consider to install CPU-version TensorFlow.')
        args.write_summary = False

    # Set random seeds
    if args.seed is not None:
        logger.info('random seed is set to %i', args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        logger.warn('random seed is not set.')

    # Chores before training
    if not os.path.exists(args.log_dir):
        logger.debug('create log directory %s', args.log_dir)
        os.makedirs(args.log_dir)

    # Initialize training
    # Define the model
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epoch, step = 0, 0
    # Load saved checkpoint
    if args.resume:
        logger.info('resuming from checkpoint %s', args.resume)
        checkpoint = torch.load(args.resume)
        logger.debug('checkpoint keys: %r', checkpoint.keys())
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        logger.info('resuming from epoch %i step %i', epoch, step)

    # GPU or not
    if len(args.gpus) > 0:
        logger.info('using GPUs %s', args.gpus)
        net = net.cuda()
        if len(args.gpus) > 1:
            net = nn.DataParallel(net, args.gpus)

    logger.info(net)
    logger.info('model parameters:')
    n_params = 0
    for name, param in net.named_parameters():
        n_params += param.nelement()
        logger.info('%s | %r | %i', name, param.size(), param.nelement())
    logger.info('# parameters: %i', n_params)

    problem = problem_registry[args.problem]()
    logger.info('# train samples: %i', problem.specs['train_size'])
    logger.info('# validation samples: %i', problem.specs['val_size'])

    train_loader = problem.get_train_loader(args.batch_size)
    val_loader = problem.get_val_loader(args.batch_size)

    loss_fn = nn.CrossEntropyLoss()

    # Summary writer
    if args.write_summary:
        writer = summary.FileWriter(args.log_dir, flush_secs=10)

    if args.n_epochs <= epoch:
        logger.warn('too few epochs to train')

    # Training loop
    while epoch < args.n_epochs:
        logger.info('starting epoch %i', epoch)
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # Build a mini-batch
            imgs, labels = batch
            if len(args.gpus) > 0:
                imgs, labels = imgs.cuda(), labels.cuda()
            imgs, labels = Variable(imgs), Variable(labels)

            preds = net(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()

            # Write training summary
            if args.write_summary:
                param_norm = global_norm(net.parameters())
                grad_norm = global_norm([param.grad for param in net.parameters()])
                summary_proto = Summary(value=[
                    Summary.Value(tag='train/loss', simple_value=loss.data[0]),
                    Summary.Value(tag='train/param_norm', simple_value=param_norm.data[0]),
                    Summary.Value(tag='train/grad_norm', simple_value=grad_norm.data[0]),
                    ])
                writer.add_summary(summary_proto, global_step=step)

            # Update parameters
            step += 1
            optimizer.step()

        # Evaluate on test
        n_correct = 0
        for batch in tqdm.tqdm(val_loader):
            imgs, labels = batch
            if len(args.gpus) > 0:
                imgs, labels = imgs.cuda(), labels.cuda()
            imgs, labels = Variable(imgs), Variable(labels)

            _, preds = net(imgs).max(1)
            n_correct += preds.eq(labels).float().sum().data[0]
        accuracy = n_correct / problem.specs['test_size']
        logger.info('epoch %i test accuracy %g', epoch, accuracy)

        # Write validation summary
        if args.write_summary:
            writer.add_summary(Summary(value=[Summary.Value(tag='test/accuracy', simple_value=accuracy)]), global_step=epoch)

        epoch += 1
        # Save training checkpoints
        checkpoint = make_checkpoint(epoch, step, optimizer, net)
        torch.save(checkpoint, os.path.join(args.log_dir, 'model_%.4f_e%i.pt' % (accuracy, epoch)))

    if args.write_summary:
        writer.close()
