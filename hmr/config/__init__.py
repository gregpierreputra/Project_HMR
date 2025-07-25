import os
from typing import Dict
from yacs.config import CfgNode as CN

"""
Config handler
Utilized in the demonstration code for accessing the model configs
"""


def get_config(
    config_file: str,
    smpl_model_path: str,
    smpl_joint_regressor_extra_path: str,
    smpl_mean_params_path: str,
) -> CN:
    """
    Read a config file and optionally merge it with the default config file.

    Args:
      config_file (str): Path to config file.

    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    # Initialize a new config node in the configurations tree
    cfg = CN(new_allowed=True)

    # Merge with the configurations defined in the demo.yaml file
    cfg.merge_from_file(config_file)

    # Utilize the SMPL model path, joint regressor extra, and mean parameters using the args path
    cfg.SMPL.MODEL_PATH = smpl_model_path
    cfg.SMPL.JOINT_REGRESSOR_EXTRA = smpl_joint_regressor_extra_path
    cfg.SMPL.MEAN_PARAMS = smpl_mean_params_path

    cfg.freeze()
    return cfg
