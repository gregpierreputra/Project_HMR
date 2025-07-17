import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        2D Keypoint Loss Module from HMR

        Arguments:
            loss_type   (str) : Toggle string between the different implemented loss functions
        """
        super(Keypoint2DLoss, self).__init__()

        # Utilized when the images utilized has 2D keypoints annotations
        if loss_type == "l1":
            self.loss_function = nn.L1Loss(reduction="none")

        # Utilized when ground truth SMPL pose parameters and shape parameters are available
        elif loss_type == "l2":
            self.loss_function = nn.MSELoss(reduction="none")

        # Additional for curiosity sake
        elif loss_type == "smooth-l1":
            self.loss_function = nn.SmoothL1Loss(reduction="none")

        else:
            raise NotImplementedError("Unimplemented loss function")

    def forward(
        self,
        projected_keypoints_2d: torch.Tensor,
        ground_truth_keypoints_2d: torch.Tensor,
    ):
        """
        Compute the 2D reprojection loss on the keypoints

        Arguments:
            projected_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints
                                                   (B: batch_size, S: num_samples, N: num_keypoints)
            ground_truth_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.

        Returns:
            torch.Tensor: 2D keypoint loss
        """
        conf = ground_truth_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]

        loss = (
            conf
            * self.loss_function(
                projected_keypoints_2d, ground_truth_keypoints_2d[:, :, :-1]
            )
        ).sum(dim=(1, 2))

        return loss.sum() / batch_size


class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        3D Keypoint Loss Module

        Arguments:
            loss_type   (str) : Toggle string between the different implemented loss functions
        """
        super(Keypoint3DLoss, self).__init__()

        # Utilized when the images utilized has 2D keypoints annotations
        if loss_type == "l1":
            self.loss_function = nn.L1Loss(reduction="none")

        # Utilized when ground truth SMPL pose parameters and shape parameters are available
        elif loss_type == "l2":
            self.loss_function = nn.MSELoss(reduction="none")

        # Additional for curiosity sake
        elif loss_type == "smooth-l1":
            self.loss_function = nn.SmoothL1Loss(reduction="none")

        else:
            raise NotImplementedError("Unimplemented loss function")

    def forward(
        self,
        projected_keypoints_3d: torch.Tensor,
        ground_truth_keypoints_3d: torch.Tensor,
        pelvis_id: int = 39,
    ):
        """
        Compute the 3D Keypoint Loss.
        Args:
            projected_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the
                                                   projected 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            ground_truth_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss
        """
        batch_size = projected_keypoints_3d.shape[0]
        ground_truth_keypoints_3d = ground_truth_keypoints_3d.clone()

        # For 3D projected keypoints, Remove 3D pelvis keypoint
        # and add additional dimension at the end to keep size as [B, 1, S, N]
        projected_keypoints_3d = projected_keypoints_3d - projected_keypoints_3d[
            :, pelvis_id, :
        ].unsqueeze(1)

        # For all 3D ground truth keypoints except the last one, remove 3D pelvis keypoint
        # and add additional dimension at the end to keep size as [B, 1, S, N]
        # Keep all keypoints except the last one
        ground_truth_keypoints_3d[:, :, :-1] = ground_truth_keypoints_3d[
            :, :, :-1
        ] - ground_truth_keypoints_3d[:, pelvis_id, :-1].unsqueeze(1)

        conf = ground_truth_keypoints_3d[:, :, -1].unsqueeze(-1).clone()

        loss = (
            conf
            * self.loss_function(
                projected_keypoints_3d, ground_truth_keypoints_3d[:, :, :-1]
            )
        ).sum(dim=(1, 2))

        return loss.sum() / batch_size


class ParameterLoss(nn.Module):
    def __init__(self):
        """
        SMPL Parameter Loss Module
        """
        super(ParameterLoss, self).__init__()
        self.loss_function = nn.MSELoss(reduction="none")

    def forward(
        self,
        projected_parameter: torch.Tensor,
        ground_truth_parameter: torch.Tensor,
        has_parameter: torch.Tensor,
    ):
        """
        Compute skinned multi-person linear (SMPL) parameter loss

        Arguments:
            projected_parameter     (torch.Tensor) : Tensor of shape [B, S, ...] containing the projected parameters (body pose / global orientation / betas)
            ground_truth_parameter  (torch.Tensor) : Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.

        Returns:
            torch.Tensor : L2 parameter loss, or also known as mean-squared error parameter loss.
        """
        # Retrieve batch size from the projected parameter tensor
        batch_size = projected_parameter.shape[0]

        # Calculate number of dimensions by taking the length of the shape of a projected parameter tensor
        number_of_dimensions = len(projected_parameter.shape)

        # Create the mask dimension by taking the batch size (e.g., 39) and inserting values of 1 based on the total number of dimensions
        mask_dimension = [batch_size] + [1] * (number_of_dimensions - 1)

        # Unpack mask dimension data, and convert it into the same type as the projected_parameter type
        has_parameter = has_parameter.type(projected_parameter.type()).view(
            *mask_dimension
        )

        loss_parameter = has_parameter * self.loss_function(
            projected_parameter, ground_truth_parameter
        )

        return loss_parameter.sum() / batch_size
