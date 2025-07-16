import os
from datetime import datetime
from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer

from hmr.datasets import DataModule
from hmr.model.hmr import HMR
from hmr.utils.misc_logger import get_logger

# Directory
_TRAIN_BASE_OUTPUT_DIRECTORY = "/home/greg/Monash_MDN_Projects/HMR_Train"
_TRAIN_MODEL_CHECKPOINT_PATH = os.path.join(
    _TRAIN_BASE_OUTPUT_DIRECTORY, "model_checkpoints"
)

# Hardcoded training variables
_TRAIN_ACCELERATOR = "cpu"
_TRAIN_MAX_EPOCH_VALUE = 100
_TRAIN_LOG_EVERY_N_STEPS = 5
_TRAIN_BATCH_SIZE = 2
_TRAIN_NUM_WORKERS = 2
_TRAIN_PREFETCH_FACTOR = 2
_TRAIN_NUM_OF_SAMPLES = 10


# Checkpoint callback hardcoded variables
_CHECKPOINT_CALLBACK_CHECKPOINT_STEPS = 2
_CHECKPOINT_CALLBACK_SAVE_TOP_K = 1

_MLFLOW_EXPERIMENT_NAME = "MDN_3D_Human_Mesh"
_MLFLOW_URI = "localhost:5000"

curr_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
_MLFLOW_RUN_NAME = f"{_MLFLOW_EXPERIMENT_NAME}-{curr_datetime}"

log = get_logger(__name__)


def train() -> Tuple[dict, dict]:
    # Setup training and validation datasets
    log.info("Instantiating data module with training and validation dataset")

    data_module = DataModule(
        training_batch_size=_TRAIN_BATCH_SIZE,
        training_number_of_workers=_TRAIN_NUM_WORKERS,
        training_prefetch_factor=_TRAIN_PREFETCH_FACTOR,
        training_number_of_samples=_TRAIN_NUM_OF_SAMPLES,
    )

    log.info("Successfully instantiated data module!")

    # Setup model
    log.info("Instantiating model {}".format("HMR"))

    model = HMR()

    # Setup MLFlow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=_MLFLOW_EXPERIMENT_NAME,
        tracking_uri=_MLFLOW_URI,
        run_name=_MLFLOW_RUN_NAME,
    )

    # Setup callbacks
    # Feel free to change any settings, this is just a base setup - Greg | 30-06-2025
    log.info("Instantiating checkpoint callback, and learning rate monitor")

    checkpoint_callback = ModelCheckpoint(
        dirpath=_TRAIN_MODEL_CHECKPOINT_PATH,
        every_n_train_steps=_CHECKPOINT_CALLBACK_CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=_CHECKPOINT_CALLBACK_SAVE_TOP_K,
    )

    callbacks = [
        checkpoint_callback,
    ]

    # Setup trainer
    # A basic trainer implementation based on what I have explored in PyTorch lightning
    log.info("Instantiating trainer <{}>".format("pytorch_lightning.Trainer"))

    trainer = Trainer(
        accelerator=_TRAIN_ACCELERATOR,
        max_epochs=_TRAIN_MAX_EPOCH_VALUE,
        log_every_n_steps=_TRAIN_LOG_EVERY_N_STEPS,
        callbacks=callbacks,
        logger=mlflow_logger,
    )

    log.info("Starting the trainer.")
    # Call the trainer
    trainer.fit(model, datamodule=data_module)
    log.info("Training done!")


def main():
    # Train the model - encapsulate in helper function for future added functionality
    train()


if __name__ == "__main__":
    main()
