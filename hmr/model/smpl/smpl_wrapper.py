import torch
import pickle
from typing import Optional

import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLHOutput

class SMPL(smplx.SMPLLayer):
    def __init__(self,
                 model_path: str,
                 mean_params: str,
                 joint_regressor_extra: Optional[str] = None,
                 gender: str = 'neutral',
                 num_body_joints: int  = 23,
                 update_hips: bool = False,
                 *args,
                 **kwargs):
        """
        Utilizing the SMPLX - an extension of the SMPL implementation that supports more joints
        
        Arguments:
            model_path          (str)  : String path to the location of the model. Example value: HMR//smpl
            mean_parameters     (str)  : String path to the location of the npz file containing the SMPL mean parameters. Example value: HMR/smpl_mean_params.npz
            joint_regressor_extra   (Optional[str]) : String path to the location of the pickle file containing the extra joint regressors. Example value: HMR/SMPL_to_J19.pkl
            gender              (str)  : String value determining the SMPL body type to use. Hardcoded to be gender 'neutral'
            num_body_joints     (int)  : Integer value determining the number of joints in the SMPL model. Hardcoded to be 23
            update_hips         (bool) : Boolean value determining whether the hip joints of the SMPL model will be updated

        Returns:
            smplx.SMPLLayer object
        """
        super(SMPL, self).__init__(model_path=model_path,
                                   gender=gender,
                                   num_body_joints=num_body_joints,
                                   mean_params=mean_params,
                                   joint_regressor_extra=joint_regressor_extra,
                                   *args,
                                   **kwargs)

        # Joint mapping values between SMPL and OpenPose
        smpl_to_openpose = [24, 12, 17, 19, 21, 
                            16, 18, 20, 0, 2, 
                            5, 8, 1, 4, 7, 
                            25, 26, 27, 28, 29, 
                            30, 31, 32, 33, 34]

        # If extra joint regressors are utilized 
        # Load the pickle file through the path and as a tensor
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        
        # Register the joint mapping as a tensor in a buffer within the module 
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, 
                *args,
                **kwargs) -> SMPLHOutput:
        """
        Run a forward pass similar to SMPL
        Append an extra set of joints if joint_regressor_extra is defined
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_mapper, :]

        # Branching logic for the hips data of SMPLX
        if self.update_hips:
            joints[:,[9,12]] = joints[:,[9,12]] + \
                0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
                0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
        
        # If additional joint regressors are present. Calculate the 3D joint location from the vertices of the regressor
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        smpl_output.joints = joints

        return smpl_output