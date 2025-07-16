from typing import Optional

import os
import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data import DataLoader

from .motion_capture_dataset import MotionCaptureDataset


def load_web_dataset(
    dataset_url: str,
    resampled: bool = True,
    shardshuffle: bool = True,
    shuffle_value: int = 2,
):
    return wds.WebDataset(
        dataset_url, resampled=resampled, shardshuffle=shardshuffle
    ).shuffle(shuffle_value)


class DataModule(pl.LightningDataModule):
    # REQUIRES CONFIGURATION
    # Base path to retrieve the cmu_mocap.npz file locally
    _hmr_training_data_base_path = "/opt/ml/data/HMR2/HMR_Train_Data"
    _hmr_eval_data_base_path = "/opt/ml/data/HMR2/HMR_Val_Data"

    # Hardcoded values for testing purposes
    _test_epoch_value = 2
    _test_shuffle_value = 2

    # -----

    # Training: Image datasets
    _mpi_inf_train_wds_url = os.path.join(
        _hmr_training_data_base_path, "mpi-inf-train-pruned/{000000..00006}.tar"
    )
    # _mpi_inf_train_wds_url = "hmr2_training_data/dataset_tars/mpi-inf-train-pruned/{000000..00006}.tar"
    _h36m_wmask_train_wds_url = (
        "hmr2_training_data/dataset_tars/h36m-train/{000000..000312}.tar"
    )
    _mpii_wmask_train_wds_url = (
        "hmr2_training_data/dataset_tars/mpii-train/{000000..000009}.tar"
    )
    _coco_2014_wmask_train_wds_url = "hmr2_training_data/dataset_tars/coco-train-2014-pruned/{000000..000017}.tar"
    _coco_2014_vitpose_replicate_pruned_train_wds_url = "hmr2_training_data/dataset_tars/coco-train-2014-vitpose-pruned/{000000..000044}.tar"
    _ava_midframes_train_wds_url = "hmr2_training_data/dataset_tars/ava-train-midframes-1fps-vitpose/{000000..000092}.tar"
    _aic_wmask_train_wds_url = (
        "hmr2_training_data/dataset_tars/aic-train-vitpose/{000000..000104}.tar"
    )
    _insta_wmask_train_wds_url = "hmr2_training_data/dataset_tars/insta-train-vitpose-replicate/{000000..003657}.tar"

    # Training: Motion capture dataset
    _cmu_mocap_train_wds_url = os.path.join(
        _hmr_training_data_base_path, "cmu_mocap.npz"
    )

    # Validation dataset
    _coco_val_wds_url = os.path.join(
        _hmr_eval_data_base_path, "coco-val/{000000..000000}.tar"
    )

    def __init__(
        self,
        training_batch_size: int = 10,
        training_number_of_workers: int = 1,
        training_prefetch_factor: int = 2,
        training_number_of_samples: int = 10,
    ) -> None:
        super().__init__()

        # Initialize dataset variables
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.motion_capture_dataset = None

        # Initialize general class variables
        self.training_batch_size = training_batch_size
        self.training_number_of_workers = training_number_of_workers
        self.training_prefetch_factor = training_prefetch_factor
        self.training_mocap_batch_size = (
            training_number_of_samples * training_batch_size
        )

    def __str__(self):
        return "Successfully loaded all of the data!"

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load the necessary training data using the WebLoader

        Although unused, stage input variable necessary to prevent a TypeError from being called due to unexpected keyword argument

        Training dataset will utilise the Ava Midframes training set
        Validation dataset will utilise the COCO validation set
        """
        if self.training_dataset == None:
            self.training_dataset = (
                load_web_dataset(self._mpi_inf_train_wds_url)
                .with_epoch(self._test_epoch_value)
                .shuffle(self._test_shuffle_value)
            )
            self.motion_capture_dataset = MotionCaptureDataset(
                self._cmu_mocap_train_wds_url
            )

            self.validation_dataset = load_web_dataset(
                self._coco_val_wds_url
            ).shuffle(self._test_shuffle_value)

    def train_dataloader(self) -> DataLoader:
        """
        Setup the training DataLoader for both images and motion capture

        Returns:
            training_dataloaders (dict) : Dictionary containing the image and motion capture dataloaders. Keys are appropriately 'img' and 'mocap'
        """
        train_dataloader = DataLoader(
            self.training_dataset,
            self.training_batch_size,
            drop_last=True,
            num_workers=self.training_number_of_workers,
            prefetch_factor=self.training_prefetch_factor,
        )
        motion_capture_dataloader = DataLoader(
            self.motion_capture_dataset,
            self.training_mocap_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.training_number_of_workers,
        )

        training_dataloaders = {
            "img": train_dataloader,
            "mocap": motion_capture_dataloader,
        }

        return training_dataloaders

    def val_dataloader(self) -> DataLoader:
        """
        Setup the validation DataLoader

        Returns:
            validation_dataloader (torch.utils.data.DataLoader): Validation dataset as a dataloader
        """
        validation_dataloader = DataLoader(
            self.validation_dataset,
            self.training_batch_size,
            drop_last=True,
            num_workers=self.training_number_of_workers,
        )

        return validation_dataloader
