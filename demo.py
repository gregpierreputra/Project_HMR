import time
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hmr import load_HMR
from hmr.datasets.vitdet_dataset import ViTDetDataset
from hmr.utils import recursive_to
from hmr.utils.renderer import Renderer, cam_crop_to_full
from hmr.utils.utils_detectron2 import DefaultPredictor_Lazy
from hmr.datasets.webdataset import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD
from hmr.model.hmr import HMRLightningModule


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def main():
    # Start time
    start = time.perf_counter()

    # Argument parser
    parser = argparse.ArgumentParser(description="HMR Demo Code")

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
        default="/opt/ml/misc/MDN/HMR2/checkpoint_file.ckpt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/input_images",
        help="Folder with input images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/opt/ml/misc/MDN/HMR2/output_images",
        help="Output folder to save rendered results",
    )
    parser.add_argument(
        "--side_view",
        dest="side_view",
        action="store_true",
        default=False,
        help="If set, render side view also",
    )
    parser.add_argument(
        "--top_view",
        dest="top_view",
        action="store_true",
        default=False,
        help="If set, render top view also",
    )
    parser.add_argument(
        "--full_frame",
        dest="full_frame",
        action="store_true",
        default=False,
        help="If set, render all people together also",
    )
    parser.add_argument(
        "--save_mesh",
        dest="save_mesh",
        action="store_true",
        default=False,
        help="If set, save meshes to disk also",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference/fitting"
    )
    parser.add_argument(
        "--file_type",
        nargs="+",
        default=["*.jpg", "*.png"],
        help="List of file extensions to consider",
    )

    args = parser.parse_args()

    # Load the model using the checkpoint
    model = HMRLightningModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        smpl_model_path=args.smpl_model_path,
        smpl_joint_regressor_extra_path=args.smpl_joint_regressor_extra_path,
        smpl_mean_params_path=args.smpl_mean_params_path,
        vitpose_backbone_pretrained_path=args.vitpose_backbone_pretrained_path,
    )

    # Setup the HMR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    # Vitdet from Detectron2
    if args.detector == "vitdet":

        from detectron2.config import LazyConfig
        import hmr

        cfg_path = (
            Path(hmr.__file__).parent / "config" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )

        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Regnety from Detectron2
    elif args.detector == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(
        focal_length_scale=model.focal_length_scale,
        image_size=model._image_size,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        faces=model.smpl.faces,
    )

    # Make the output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Retrieve all demonstration images that end with .jpg or .png
    img_paths = [
        img for end in args.file_type for img in Path(args.img_folder).glob(end)
    ]

    # Iterate over all images in the folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in the image
        time_det_start = time.perf_counter()
        det_out = detector(img_cv2)
        time_det = time.perf_counter() - time_det_start
        print(f"DET FORWARD TIME {time_det}")

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(
            img_size=model._image_size,
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
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                time_model_start = time.perf_counter()
                out = model(batch)
                time_forward = time.perf_counter() - time_model_start
                print(f"HMR FORWARD TIME: {time_forward}")

            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                model.focal_length_scale / model._image_size * img_size.max()
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
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch["personid"][n])
                white_img = (
                    torch.ones_like(batch["img"][n]).cpu()
                    - DEFAULT_MEAN[:, None, None] / 255
                ) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = batch["img"][n].cpu() * (
                    DEFAULT_STD[:, None, None] / 255
                ) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                time_start_render = time.perf_counter()
                regression_img = renderer(
                    out["pred_vertices"][n].detach().cpu().numpy(),
                    out["pred_cam_t"][n].detach().cpu().numpy(),
                    batch["img"][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                time_render = time.perf_counter() - time_start_render
                print(f"RENDER TIME: {time_render}")

                final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(
                        out["pred_vertices"][n].detach().cpu().numpy(),
                        out["pred_cam_t"][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                    )
                    final_img = np.concatenate([final_img, side_img], axis=1)

                if args.top_view:
                    top_img = renderer(
                        out["pred_vertices"][n].detach().cpu().numpy(),
                        out["pred_cam_t"][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        top_view=True,
                    )
                    final_img = np.concatenate([final_img, top_img], axis=1)

                cv2.imwrite(
                    os.path.join(args.out_folder, f"{img_fn}_{person_id}.png"),
                    255 * final_img[:, :, ::-1],
                )

                # Add all verts and cams to list
                verts = out["pred_vertices"][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(
                        verts, camera_translation, LIGHT_BLUE
                    )
                    tmesh.export(
                        os.path.join(args.out_folder, f"{img_fn}_{person_id}.obj")
                    )

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args
            )

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

            cv2.imwrite(
                os.path.join(args.out_folder, f"{img_fn}_all.png"),
                255 * input_img_overlay[:, :, ::-1],
            )

        end = time.perf_counter()
        print(end - start)


if __name__ == "__main__":
    main()
