import logging

# Set up logging format
log_format = '[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
logging.basicConfig(format=log_format)
logger = logging.getLogger('util')
logger.setLevel(logging.INFO)

# Dynamic import of torch tensors
import torch
if torch.cuda.is_available():
    logger.info('CUDA devices are available')
    import torch.cuda as tt
else:
    logger.info('CUDA devices are not available')
    import torch as tt

def make_checkpoint(epoch, step, optimizer, model, extra={}):
    '''Save a dictionary containing complete training state'''
    return {
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'extra': extra,
    }
