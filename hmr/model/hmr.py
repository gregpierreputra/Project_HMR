from pathlib import Path
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch

from hmr.utils.geometry import aa_to_rotmat, perspective_projection
from hmr.model.backbone.vitpose import ViTBackbone
from hmr.model.discriminator import Discriminator
from hmr.model.losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss
from hmr.model.smpl.smpl_wrapper import SMPL
from hmr.model.smpl_head.transformer import SMPLTransformerDecoderHead


class HMRLightningModule(pl.LightningModule):
    def __init__(
        self,
        smpl_model_path: str,
        smpl_mean_params_path: str,
        smpl_joint_regressor_extra_path: str,
        vitpose_backbone_pretrained_path: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        grad_clip_val: float = 1.0,
        focal_length_scale: int = 5000,
        loss_3d_keypoint_weight: float = 0.05,
        loss_2d_keypoint_weight: float = 0.01,
        loss_global_orient_weight: float = 0.001,
        loss_body_pose_weight: float = 0.001,
        loss_betas_weight: float = 0.0005,
        loss_adversarial_weight: float = 0.0005,
    ):
        """
        Setup the HMR module
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.focal_length_scale = focal_length_scale
        self.loss_3d_keypoint_weight = loss_3d_keypoint_weight
        self.loss_2d_keypoint_weight = loss_2d_keypoint_weight
        self.loss_smpl_weight_dict = {
            "GLOBAL_ORIENT": loss_global_orient_weight,
            "BODY_POSE": loss_body_pose_weight,
            "BETAS": loss_betas_weight,
            "ADVERSARIAL": loss_adversarial_weight,
        }

        self._image_size = 256

        # Load the ViTBackbone state dictionary
        # Utilize the weights for ViTPose Small model extracted by Agi
        vitpose_state_dict = torch.load(
            vitpose_backbone_pretrained_path, map_location="cpu"
        )

        # Create the ViTBackbone feature extractor
        # Matching the variable values to those utilized in HMR2
        self.backbone = ViTBackbone(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.1,
        )

        # Load the pretrained weights for the ViTBackbone
        self.backbone.load_state_dict(vitpose_state_dict, strict=True)

        # Create SMPL head
        self.smpl_head = SMPLTransformerDecoderHead(
            smpl_mean_params_path=smpl_mean_params_path
        )

        # Create discriminator
        self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type="l1")
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type="l1")
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        self.smpl = SMPL(
            model_path=smpl_model_path,
            joint_regressor_extra=smpl_joint_regressor_extra_path,
        )

        # Disabling automatic optimization due to use of adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        """
        Helper function to get all the parameters for the SMPLTransformerDecoderHead and the ViTBackbone for the optimizer
        """
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())

        return all_params

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup the Optimizers for the model and discriminator

        Returns:
            Tuple of torch.optim.Optimizer for the model and discriminator respectively
        """
        param_groups = [
            {
                "params": filter(lambda p: p.requires_grad, self.get_parameters()),
                "lr": self.learning_rate,
            }
        ]

        optimizer_model = torch.optim.AdamW(
            params=param_groups, weight_decay=self.weight_decay
        )

        optimizer_discriminator = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer_model, optimizer_discriminator

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network

        Although unused, the train variable is necessary to prevent a TypeError from being thrown during training

        Arguments:
            batch (Dict)    : Dictionary containing batch data

        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch["img"]
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        conditioning_features = self.backbone(x[:, :, :, 32:-32])

        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_features)

        # Store useful regression outputs to the output dictionary
        output = {}
        output["pred_cam"] = pred_cam
        output["pred_smpl_params"] = {k: v.clone() for k, v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params["body_pose"].device
        dtype = pred_smpl_params["body_pose"].dtype

        # Determine focal length based on predicted SMPL parameters
        focal_length = self.focal_length_scale * torch.ones(
            batch_size, 2, device=device, dtype=dtype
        )

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self._image_size * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        # Save the pred_cam_t and focal_length to the output dictionary
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length

        # Compute the model vertices, joints, and the projected joints
        # Retrieve global orientation, body pose, betas and reshape
        pred_smpl_params["global_orient"] = pred_smpl_params["global_orient"].reshape(
            batch_size, -1, 3, 3
        )
        pred_smpl_params["body_pose"] = pred_smpl_params["body_pose"].reshape(
            batch_size, -1, 3, 3
        )
        pred_smpl_params["betas"] = pred_smpl_params["betas"].reshape(batch_size, -1)

        # Retrieve the SMPLX model outputs and predictions
        smpl_output = self.smpl(
            **{k: v.float() for k, v in pred_smpl_params.items()},
            pose2rot=False,
        )
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t,
            focal_length=focal_length / self._image_size,
        )
        output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        return output

    def compute_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output

        Arguments:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
        Returns:
            torch.Tensor : Total loss for current batch
        """
        # Retrieve the different values from the output dictionary
        pred_smpl_params = output["pred_smpl_params"]
        pred_keypoints_2d = output["pred_keypoints_2d"]
        pred_keypoints_3d = output["pred_keypoints_3d"]

        batch_size = pred_smpl_params["body_pose"].shape[0]

        # Retrieve annotations from the batch dictionary
        gt_keypoints_2d = batch["keypoints_2d"]
        gt_keypoints_3d = batch["keypoints_3d"]
        gt_smpl_params = batch["smpl_params"]
        has_smpl_params = batch["has_smpl_params"]
        is_axis_angle = batch["smpl_params_is_axis_angle"]

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(
            pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25 + 14
        )

        # Compute loss on SMPL parameters
        loss_smpl_params = {}

        # For every predicted SMPL parameter (body pose, shape, camera)
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)

            # Converts an axis-angle representation to a rotation matrix by first converting it to a quaternion
            # Goes from Tensor of shape (B, 3) to rotation matrices with shape (B, 3, 3)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)

            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(
                pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt
            )

        # Calculate the overall loss taking into consideration the weights we put on the different metrics
        loss = (
            self.loss_3d_keypoint_weight * loss_keypoints_3d
            + self.loss_2d_keypoint_weight * loss_keypoints_2d
            + sum(
                [
                    loss_smpl_params[k] * self.loss_smpl_weight_dict[k.upper()]
                    for k in loss_smpl_params
                ]
            )
        )

        # Save all of the losses into a dictionary
        losses = dict(
            loss=loss.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach(),
            loss_keypoints_3d=loss_keypoints_3d.detach(),
        )

        # Add the losses from the SMPL parameters into the losses dictionary
        for k, v in loss_smpl_params.items():
            losses["loss_" + k] = v.detach()

        output["losses"] = losses

        return loss

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(
        self,
        batch: Dict,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch["body_pose"]
        gt_betas = batch["betas"]

        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())

        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)

        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real

        loss = self.loss_smpl_weight_dict["ADVERSARIAL"] * loss_disc

        optimizer.zero_grad()

        self.manual_backward(loss)

        optimizer.step()

        return loss_disc.detach()

    def training_step(self, joint_batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch["img"]
        mocap_batch = joint_batch["mocap"]

        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.loss_smpl_weight_dict["ADVERSARIAL"] > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch["img"].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output["pred_smpl_params"]

        # if self.cfg.get('UPDATE_GT_SPIN', False):
        #     self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output)

        if self.loss_smpl_weight_dict["ADVERSARIAL"] > 0:
            disc_out = self.discriminator(
                pred_smpl_params["body_pose"].reshape(batch_size, -1),
                pred_smpl_params["betas"].reshape(batch_size, -1),
            )
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.loss_smpl_weight_dict["ADVERSARIAL"] * loss_adv

        optimizer.zero_grad()

        self.manual_backward(loss)

        # Clip gradient
        if self.grad_clip_val > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                self.get_parameters(),
                self.grad_clip_val,
                error_if_nonfinite=True,
            )
            self.log(
                "train-grad_norm",
                gn,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        optimizer.step()

        if self.loss_smpl_weight_dict["ADVERSARIAL"] > 0:
            loss_disc = self.training_step_discriminator(
                mocap_batch,
                pred_smpl_params["body_pose"].reshape(batch_size, -1),
                pred_smpl_params["betas"].reshape(batch_size, -1),
                optimizer_disc,
            )
            output["losses"]["loss_gen"] = loss_adv
            output["losses"]["loss_disc"] = loss_disc

        self.log(
            "train-loss",
            output["losses"]["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train-loss_kp3d",
            output["losses"]["loss_keypoints_3d"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train-loss_adv",
            output["losses"]["loss_gen"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train-loss_disc",
            output["losses"]["loss_disc"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return output

    def validation_step(self, batch: Dict) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output)

        batch_size = batch["img"]

        self.log("val-loss", loss, on_epoch=True, batch_size=batch_size)

        output["loss"] = loss

        return output


def _test():
    base_dir = Path("/opt/ml/misc/MDN/HMR2")
    vitpose_backbone_pretrained_path = str(base_dir / "vitpose_small_backbone.pth")

    # Can be obtained on the following link
    # SMPL Model: https://smplify.is.tue.mpg.de/download.php
    #   (SMPLIFY_CODE_V2.ZIP or mpips_smplify_public_v2.zip)
    # SMPL parameters and regressors: https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_data.tar.gz
    smpl_model_path = str(
        base_dir / "mpips_smplify_public_v2/smplify_public/code/models"
    )
    smpl_mean_params_path = str(base_dir / "smpl_mean_params.npz")
    smpl_joint_regressor_extra_path = str(base_dir / "SMPL_to_J19.pkl")

    hmr_lm = HMRLightningModule(
        smpl_model_path=smpl_model_path,
        smpl_mean_params_path=smpl_mean_params_path,
        smpl_joint_regressor_extra_path=smpl_joint_regressor_extra_path,
        vitpose_backbone_pretrained_path=vitpose_backbone_pretrained_path,
        learning_rate=1e-5,
    )


if __name__ == "__main__":
    _test()
