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
