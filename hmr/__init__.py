import os
from pathlib import Path

from hmr.model.hmr import HMRLightningModule
from hmr.config import get_config

def check_smpl_exists(smpl_file_location: str):
    smpl_model_exists = os.path.exists(smpl_file_location)

    if not smpl_model_exists:
        raise FileNotFoundError("Did not find SMPL model in location: {}. Please dowmnload it from https://smplify.is.tue.mpg.de/ and place it in the appropriate location"
                                .format(smpl_file_location))

    return True

def load_HMR(checkpoint_path: str,
             smpl_file_location: str,
             smpl_model_path: str,
             smpl_mean_params_path: str,
             smpl_joint_regressor_extra_path: str,
             vitpose_backbone_pretrained_path: str):

    # Check to ensure that an SMPL model exists
    check_smpl_exists(smpl_file_location)

    # Load the model utilizing the checkpoint path defined in the argument parser
    model = HMRLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False, 
        smpl_model_path=smpl_model_path,
        smpl_joint_regressor_extra_path=smpl_joint_regressor_extra_path,
        smpl_mean_params_path=smpl_mean_params_path,
        vitpose_backbone_pretrained_path=vitpose_backbone_pretrained_path
        )

    return model