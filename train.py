from datetime import datetime
from typing import Tuple
from dataclasses import dataclass, asdict
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer

from hmr.datasets import HMRDataModule
from hmr.model.hmr import HMRLightningModule
from hmr.utils.misc_logger import get_logger


@dataclass
class TrainArgument:
    train_dataset_name: str
    val_dataset_name: str
    mocap_datafile_path: str
    amass_poses_hist100_path: str
    smpl_model_path: str
    smpl_mean_params_path: str
    smpl_joint_regressor_extra_path: str
    vitpose_backbone_pretrained_path: str
    checkpoint_path: str
    accelerator: str
    max_epochs: int
    log_every_n_steps: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    mocap_num_train_samples: int
    checkpoint_topk: int
    mlflow_experiment_name: str
    mlflow_uri: str
    mlflow_run_name: str = ""
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_val: float = 1.0
    focal_length_scale: int = 5000
    loss_3d_keypoint_weight: float = 0.05
    loss_2d_keypoint_weight: float = 0.01
    loss_global_orient_weight: float = 0.001
    loss_body_pose_weight: float = 0.001
    loss_betas_weight: float = 0.0005
    loss_adversarial_weight: float = 0.0005


def _cli_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default="COCO-TRAIN-2014-WMASK-PRUNED",
        help="Name of the training dataset. See hmr/datasets/__init__.py for more information.",
    )
    parser.add_argument(
        "--val_dataset_name",
        type=str,
        default="COCO-VAL",
        help="Name of the val dataset. See hmr/datasets/__init__.py for more information.",
    )
    parser.add_argument(
        "--mocap_datafile_path",
        type=str,
        default="/opt/ml/data/MDN/hmr2_training_data/cmu_mocap.npz",
        help="Path for mocap dataset",
    )
    parser.add_argument(
        "--amass_poses_hist100_path",
        type=str,
        default="/opt/ml/data/MDN/hmr2_training_data/amass_poses_hist100_SMPL+H_G.npy",
        help="Path to amass poses dataset",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/mpips_smplify_public_v2/smplify_public/code/models",
        help="Path to the SMPL model",
    )
    parser.add_argument(
        "--smpl_mean_params_path",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/smpl_mean_params.npz",
        help="Path to SMPL mean parameters",
    )
    parser.add_argument(
        "--smpl_joint_regressor_extra_path",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/SMPL_to_J19.pkl",
        help="Path to extra joint regressor file",
    )
    parser.add_argument(
        "--vitpose_backbone_pretrained_path",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/vitpose_small_backbone.pth",
        help="Path to pretrained ViTPose backbone",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or gpu)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="How often to log training info",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--mocap_num_train_samples",
        type=int,
        default=10,
        help="Number of motion capture training samples to use on each batch",
    )
    parser.add_argument(
        "--checkpoint_topk",
        type=int,
        default=1,
        help="Number of top model checkpoints to keep",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="hmr",
        help="Name of the MLflow experiment",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="http://localhost:5000",
        help="URI of the MLflow tracking server",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default="",
        help="Optional name of the MLflow run",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--grad_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--focal_length_scale", type=int, default=5000, help="Camera focal length scale"
    )
    parser.add_argument(
        "--loss_3d_keypoint_weight",
        type=float,
        default=0.05,
        help="Weight for 3D keypoint loss",
    )
    parser.add_argument(
        "--loss_2d_keypoint_weight",
        type=float,
        default=0.01,
        help="Weight for 2D keypoint loss",
    )
    parser.add_argument(
        "--loss_global_orient_weight",
        type=float,
        default=0.001,
        help="Weight for global orientation loss",
    )
    parser.add_argument(
        "--loss_body_pose_weight",
        type=float,
        default=0.001,
        help="Weight for body pose loss",
    )
    parser.add_argument(
        "--loss_betas_weight", type=float, default=0.0005, help="Weight for betas loss"
    )
    parser.add_argument(
        "--loss_adversarial_weight",
        type=float,
        default=0.0005,
        help="Weight for adversarial loss",
    )

    args = parser.parse_args(namespace=TrainArgument)

    if not args.mlflow_run_name:
        curr_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        args.mlflow_run_name = f"{args.mlflow_experiment_name}-{curr_datetime}"

    return args


def train(train_args: TrainArgument) -> Tuple[dict, dict]:
    log = get_logger(__name__)

    # Setup training and validation datasets
    log.info("Instantiating data module with training and validation dataset")

    data_module = HMRDataModule(
        train_dataset_name=train_args.train_dataset_name,
        val_dataset_name=train_args.val_dataset_name,
        mocap_datafile_path=train_args.mocap_datafile_path,
        amass_poses_hist100_path=train_args.amass_poses_hist100_path,
        batch_size=train_args.batch_size,
        num_workers=train_args.batch_size,
        prefetch_factor=train_args.prefetch_factor,
        mocap_num_train_samples=train_args.mocap_num_train_samples,
    )

    log.info("Successfully instantiated data module!")

    # Setup model
    log.info("Instantiating HMRLightningModule")

    model = HMRLightningModule(
        smpl_model_path=train_args.smpl_model_path,
        smpl_mean_params_path=train_args.smpl_mean_params_path,
        smpl_joint_regressor_extra_path=train_args.smpl_joint_regressor_extra_path,
        vitpose_backbone_pretrained_path=train_args.vitpose_backbone_pretrained_path,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        grad_clip_val=train_args.grad_clip_val,
        focal_length_scale=train_args.focal_length_scale,
        loss_3d_keypoint_weight=train_args.loss_3d_keypoint_weight,
        loss_2d_keypoint_weight=train_args.loss_2d_keypoint_weight,
        loss_global_orient_weight=train_args.loss_global_orient_weight,
        loss_body_pose_weight=train_args.loss_body_pose_weight,
        loss_betas_weight=train_args.loss_betas_weight,
        loss_adversarial_weight=train_args.loss_adversarial_weight,
    )

    # Setup MLFlow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=train_args.mlflow_experiment_name,
        tracking_uri=train_args.mlflow_uri,
        run_name=train_args.mlflow_run_name,
    )

    # Setup callbacks
    # Feel free to change any settings, this is just a base setup - Greg | 30-06-2025
    log.info("Instantiating checkpoint callback, and learning rate monitor")

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_args.checkpoint_path,
        save_last=True,
        save_top_k=train_args.checkpoint_topk,
        monitor="train-loss",
        mode="min",
    )

    callbacks = [
        checkpoint_callback,
    ]

    # Setup trainer
    # A basic trainer implementation based on what I have explored in PyTorch lightning
    log.info("Instantiating trainer <pytorch_lightning.Trainer>")

    trainer = Trainer(
        accelerator=train_args.accelerator,
        max_epochs=train_args.max_epochs,
        log_every_n_steps=train_args.log_every_n_steps,
        callbacks=callbacks,
        logger=mlflow_logger,
    )

    log.info("Starting the trainer.")

    # Log Hyperparameters
    mlflow_logger.log_hyperparams(asdict(train_args))

    # Call the trainer
    trainer.fit(model, datamodule=data_module)
    log.info("Training done!")


def main():
    train_args = _cli_parser()
    train(train_args)


if __name__ == "__main__":
    main()
