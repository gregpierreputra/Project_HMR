import logging

from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:

    # Unnecessary to add the different logging levels and use a rank_zero_only wrapper
    # At most will utilize single GPU training. Could extend further if utilizing multi GPU setup
    logger = logging.getLogger(name)

    return logger
