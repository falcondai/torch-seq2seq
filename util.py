import logging

# Set up logging format
log_format = '[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
logging.basicConfig(format=log_format)
logger = logging.getLogger('util')
logger.setLevel(logging.INFO)

import torch

def make_checkpoint(epoch, step, optimizer, model, extra={}):
    '''Save a dictionary containing complete training state'''
    return {
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'extra': extra,
    }

def global_norm(parameters):
    square = 0
    for parameter in parameters:
        square += parameter.norm() ** 2
    return square.sqrt()
