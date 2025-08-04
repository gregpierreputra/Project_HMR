import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from hmr.datasets.vitdet_dataset import ViTDetDataset
from hmr.datasets.webdataset import DEFAULT_MEAN, DEFAULT_STD
from hmr.model.hmr import HMRLightningModule
from hmr.utils import recursive_to
from hmr.utils.renderer import Renderer, cam_crop_to_full
from hmr.utils.utils_detectron2 import DefaultPredictor_Lazy
from hmr.model.detection import YolosDetector

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


class HMRDemo:
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if self.device.type == "cuda":
            print("Using CUDA")

        # Load the model using the checkpoint
        self.model = HMRLightningModule.load_from_checkpoint(
            checkpoint_path=self.args.checkpoint,
            smpl_model_path=self.args.smpl_model_path,
            smpl_joint_regressor_extra_path=self.args.smpl_joint_regressor_extra_path,
            smpl_mean_params_path=self.args.smpl_mean_params_path,
            vitpose_backbone_pretrained_path=self.args.vitpose_backbone_pretrained_path,
        )

        # Setup the HMR model
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load detector
        if self.args.detector == "vitdet":
            # Vitdet from Detectron2
            from detectron2.config import LazyConfig
            import hmr

            cfg_path = (
                Path(hmr.__file__).parent
                / "config"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )

            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = (
                    0.25
                )

            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif self.args.detector == "regnety":
            # Regnety from Detectron2
            from detectron2 import model_zoo

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif self.args.detector == "yolos":
            self.detector = YolosDetector(use_cuda=(self.device.type == "cuda"))

        # Setup the renderer
        self.renderer = Renderer(
            focal_length_scale=self.model.focal_length_scale,
            image_size=self.model._image_size,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
            faces=self.model.smpl.faces,
        )

        self._obj_savepath = "demo/output.obj"

    @torch.inference_mode()
    def run(self, img_input: str | np.ndarray):
        # Iterate over all images in the folder
        if isinstance(img_input, (str, Path)):
            img_cv2 = cv2.imread(str(img_input))
        else:
            img_cv2 = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)

        # Detect humans in the image
        det_out = self.detector(img_cv2)

        if self.args.detector != "yolos":
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        else:
            boxes = det_out.cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(
            img_size=self.model._image_size,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
            img_cv2=img_cv2,
            boxes=boxes,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )

        all_verts = []
        all_cam_t = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            out = self.model(batch)

            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model.focal_length_scale / self.model._image_size * img_size.max()
            )
            pred_cam_t_full = (
                cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                )
                .detach()
                .cpu()
                .numpy()
            )

            # Render the result
            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                person_id = int(batch["personid"][n])

                # Add all verts and cams to list
                verts = out["pred_vertices"][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                if person_id == dataset.max_area_index:
                    camera_translation = cam_t.copy()
                    tmesh = self.renderer.vertices_to_trimesh(
                        verts, camera_translation, LIGHT_BLUE
                    )
                    tmesh.export(self._obj_savepath)

        if len(all_verts) < 1:
            return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # Render front view
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        cam_view = self.renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=img_size[-1], **misc_args
        )

        # Overlay image
        input_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_img /= 255.0

        # Add alpha channel
        input_img = np.concatenate(
            [input_img, np.ones_like(input_img[:, :, :1])], axis=-1
        )

        input_img_overlay = (input_img[:, :, :3] * (1 - cam_view[:, :, 3:])) + (
            cam_view[:, :, :3] * cam_view[:, :, 3:]
        )

        input_img_overlay = np.astype(input_img_overlay * 255, np.uint8)

        return input_img_overlay, self._obj_savepath

    def __call__(self, img_input):
        return self.run(img_input)


def _parse_cli_args(with_image_path=True):
    # Argument parser
    parser = argparse.ArgumentParser(description="HMR Demo Code")

    if with_image_path:
        parser.add_argument(
            "image_path",
            type=str,
            help="Path to the image file",
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
        "--checkpoint",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/checkpoints/hmr-18_07_2025_13_10_32/epoch_0086-step_000086130.ckpt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolos",
        choices=["vitdet", "regnety", "yolos"],
        help="Using regnety improves runtime",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference/fitting"
    )

    args = parser.parse_args()

    return args


def _main():
    from PIL import Image

    args = _parse_cli_args()

    image_path = args.image_path

    hmr_demo = HMRDemo(args=args)
    render, _ = hmr_demo(image_path)

    Image.fromarray(render).save("demo/output.png")


if __name__ == "__main__":
    _main()
