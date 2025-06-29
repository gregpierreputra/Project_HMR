import os
from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from src.datasets import DataModule
from src.utils.misc_logger import get_logger

# Directory
_OUTPUT_DIRECTORY = ""

# General variables
_GENERAL_CHECKPOINT_STEPS = 5000
_GENERAL_SAVE_TOP_K_MODEL = 1

log = get_logger(__name__)

def train() -> Tuple[dict, dict]:
    
    # Setup training and validation datasets
    log.info("Instantiating data module with training and validation dataset")
    
    data_module = DataModule()

    # Setup model
    log.info("Instantiating model {}".format(""))
    
    model = ""

    # Setup Tensorboard logger
    log.info("Instantiating Tensorboard logger for output location: {}".format(
        os.path.join(_OUTPUT_DIRECTORY, 'tensorboard')
    ))

    logger = TensorBoardLogger(
        os.path.join(_OUTPUT_DIRECTORY, 'tensorboard'),
        name='',
        version='',
        default_hp_metric=False)

    # Setup checkpoint saving
    # Feel free to change any settings, this is just a base setup - Greg | 30-06-2025 
    log.info("Instantiating checkpoint callback, and learning rate monitor")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            _OUTPUT_DIRECTORY,
            'checkpoints'),
        every_n_train_steps=_GENERAL_CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=_GENERAL_SAVE_TOP_K_MODEL)
    
    learning_rate_monitor = RichProgressBar()
    
    callbacks = [
        checkpoint_callback,
        learning_rate_monitor
    ]

    # Setup trainer
    log.info("Instantiating trainer <{}>".format(
        "pytorch_lightning.Trainer"))
    
    # Setup hyperparameter logging

    # Train the model