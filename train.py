import os
from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from hmr.datasets import DataModule
from hmr.utils.misc_logger import get_logger

# Directory
_OUTPUT_DIRECTORY = ""

# Checkpoint callback hardcoded variables
_CHECKPOINT_CALLBACK_CHECKPOINT_STEPS = 5000
_CHECKPOINT_CALLBACK_SAVE_TOP_K_MODEL = 2

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

    # Setup callbacks
    # Feel free to change any settings, this is just a base setup - Greg | 30-06-2025 
    log.info("Instantiating checkpoint callback, and learning rate monitor")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            _OUTPUT_DIRECTORY,
            'checkpoints'),
        filename="{epoch}-{val_dice:.2f}",
        every_n_train_steps=_CHECKPOINT_CALLBACK_CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=_CHECKPOINT_CALLBACK_SAVE_TOP_K_MODEL)
    
    learning_rate_monitor = RichProgressBar()
    
    callbacks = [
        checkpoint_callback,
        learning_rate_monitor
    ]

    # Setup trainer
    # A basic trainer implementation based on what I have explored in PyTorch lightning
    log.info("Instantiating trainer <{}>".format(
        "pytorch_lightning.Trainer"))
    
    trainer = Trainer(
        accelerator="",
        max_epochs="",
        logger="",
        log_every_n_steps="",
        callbacks=callbacks
    )

    # Call the trainer
    trainer.fit(model, datamodule=data_module)

def main():
    # Train the model - encapsulate in helper function for future added functionality
    train()

if __name__ == "__main__":
    main()