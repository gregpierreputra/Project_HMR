from typing import List
from pathlib import Path

import cv2
import webdataset as wds
import numpy as np
import torch
import braceexpand
from webdataset.compat import WebDataset

from hmr.datasets.utils import get_example, expand_to_aspect_ratio
from hmr.datasets.smplh_prob_filter import poses_check_probable, load_amass_hist_smooth


_AIC_TRAIN_CORRUPT_KEYS = {
    "0a047f0124ae48f8eee15a9506ce1449ee1ba669",
    "1a703aa174450c02fbc9cfbf578a5435ef403689",
    "0394e6dc4df78042929b891dbc24f0fd7ffb6b6d",
    "5c032b9626e410441544c7669123ecc4ae077058",
    "ca018a7b4c5f53494006ebeeff9b4c0917a55f07",
    "4a77adb695bef75a5d34c04d589baf646fe2ba35",
    "a0689017b1065c664daef4ae2d14ea03d543217e",
    "39596a45cbd21bed4a5f9c2342505532f8ec5cbb",
    "3d33283b40610d87db660b62982f797d50a7366b",
}

_CORRUPT_KEYS = {
    *{f"aic-train/{k}" for k in _AIC_TRAIN_CORRUPT_KEYS},
    *{f"aic-train-vitpose/{k}" for k in _AIC_TRAIN_CORRUPT_KEYS},
}

_BODY_PERMUTATION = [
    0,
    1,
    5,
    6,
    7,
    2,
    3,
    4,
    8,
    12,
    13,
    14,
    9,
    10,
    11,
    16,
    15,
    18,
    17,
    22,
    23,
    24,
    19,
    20,
    21,
]
_EXTRA_PERMUTATION = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
_FLIP_KEYPOINT_PERMUTATION = _BODY_PERMUTATION + [25 + i for i in _EXTRA_PERMUTATION]

DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256


def expand_urls(urls: str | List[str]) -> List[str]:
    """Expand path(s) with braces notation to list of paths.

    Args:
        urls (str | List[str]): The path(s) to be expanded.

    Raises:
        FileNotFoundError: When any of the individual files does not exist.

    Returns:
        List[str]: List of individual file paths
    """
    if isinstance(urls, str):
        urls = [urls]

    expanded_urls = []
    for url in urls:
        for u in braceexpand.braceexpand(url):
            if not Path(u).exists():
                raise FileNotFoundError(u)
            expanded_urls.append(u)
    return expanded_urls


def load_tars_as_webdataset(
    urls: str | List[str],
    train: bool,
    amass_poses_hist100_path: str,
    resampled=False,
    epoch_size=None,
    cache_dir=None,
    shuffle_size=4000,
    SCALE_FACTOR: float = 0.3,
    ROT_FACTOR: int = 30,
    TRANS_FACTOR: float = 0.02,
    COLOR_SCALE: float = 0.2,
    ROT_AUG_RATE: float = 0.6,
    DO_FLIP: bool = True,
    FLIP_AUG_RATE: float = 0.5,
    EXTREME_CROP_AUG_RATE: float = 0.10,
    EXTREME_CROP_AUG_LEVEL: float = 1,
    **kwargs,
) -> WebDataset:
    """
    Loads the dataset from a webdataset tar file.
    """

    IMG_SIZE = 256
    BBOX_SHAPE = None
    MEAN = DEFAULT_MEAN
    STD = DEFAULT_STD

    def split_data(source):
        for item in source:
            datas = item["data.pyd"]
            for data in datas:
                if "detection.npz" in item:
                    det_idx = data["extra_info"]["detection_npz_idx"]
                    mask = item["detection.npz"]["masks"][det_idx]
                else:
                    mask = np.ones_like(item["jpg"][:, :, 0], dtype=bool)
                yield {
                    "__key__": item["__key__"],
                    "jpg": item["jpg"],
                    "data.pyd": data,
                    "mask": mask,
                }

    def suppress_bad_kps(item, thresh=0.0):
        if thresh > 0:
            kp2d = item["data.pyd"]["keypoints_2d"]
            kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
            item["data.pyd"]["keypoints_2d"] = np.concatenate(
                [kp2d[:, :2], kp2d_conf[:, None]], axis=1
            )
        return item

    def filter_numkp(item, numkp=4, thresh=0.0):
        kp_conf = item["data.pyd"]["keypoints_2d"][:, 2]
        return (kp_conf > thresh).sum() > numkp

    def filter_reproj_error(item, thresh=10**4.5):
        losses = (
            item["data.pyd"]
            .get("extra_info", {})
            .get("fitting_loss", np.array({}))
            .item()
        )
        reproj_loss = losses.get("reprojection_loss", None)
        return reproj_loss is None or reproj_loss < thresh

    def filter_bbox_size(item, thresh=1):
        bbox_size_min = item["data.pyd"]["scale"].min().item() * 200.0
        return bbox_size_min > thresh

    def filter_no_poses(item):
        return item["data.pyd"]["has_body_pose"] > 0

    def supress_bad_betas(item, thresh=3):
        has_betas = item["data.pyd"]["has_betas"]
        if thresh > 0 and has_betas:
            betas_abs = np.abs(item["data.pyd"]["betas"])
            if (betas_abs > thresh).any():
                item["data.pyd"]["has_betas"] = False
        return item

    amass_poses_hist100_smooth = load_amass_hist_smooth(amass_poses_hist100_path)

    def supress_bad_poses(item):
        has_body_pose = item["data.pyd"]["has_body_pose"]
        if has_body_pose:
            body_pose = item["data.pyd"]["body_pose"]
            pose_is_probable = poses_check_probable(
                torch.from_numpy(body_pose)[None, 3:], amass_poses_hist100_smooth
            ).item()
            if not pose_is_probable:
                item["data.pyd"]["has_body_pose"] = False
        return item

    def poses_betas_simultaneous(item):
        # We either have both body_pose and betas, or neither
        has_betas = item["data.pyd"]["has_betas"]
        has_body_pose = item["data.pyd"]["has_body_pose"]
        item["data.pyd"]["has_betas"] = item["data.pyd"]["has_body_pose"] = np.array(
            float((has_body_pose > 0) and (has_betas > 0))
        )
        return item

    def set_betas_for_reg(item):
        # Always have betas set to true
        has_betas = item["data.pyd"]["has_betas"]
        betas = item["data.pyd"]["betas"]

        if not (has_betas > 0):
            item["data.pyd"]["has_betas"] = np.array(float((True)))
            item["data.pyd"]["betas"] = betas * 0
        return item

    # Load the dataset
    if epoch_size is not None:
        resampled = True

    def corrupt_filter(sample):
        return sample["__key__"] not in _CORRUPT_KEYS

    shardshuffle = 100 if train else False

    dataset: WebDataset = WebDataset(
        expand_urls(urls),
        nodesplitter=wds.split_by_node,
        shardshuffle=shardshuffle,
        resampled=resampled,
        cache_dir=cache_dir,
    ).select(corrupt_filter)
    if train:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.decode("rgb8").rename(jpg="jpg;jpeg;png")

    # Process the dataset
    dataset = dataset.compose(split_data)

    # Filter/clean the dataset
    SUPPRESS_KP_CONF_THRESH = 0.3
    SUPPRESS_BETAS_THRESH = 3.0
    SUPPRESS_BAD_POSES = True
    POSES_BETAS_SIMULTANEOUS = True
    BETAS_REG = True
    FILTER_NO_POSES = False
    FILTER_NUM_KP = 4
    FILTER_NUM_KP_THRESH = 0.0
    FILTER_REPROJ_THRESH = 31000
    FILTER_MIN_BBOX_SIZE = 0.0
    if SUPPRESS_KP_CONF_THRESH > 0:
        dataset = dataset.map(
            lambda x: suppress_bad_kps(x, thresh=SUPPRESS_KP_CONF_THRESH)
        )
    if SUPPRESS_BETAS_THRESH > 0:
        dataset = dataset.map(
            lambda x: supress_bad_betas(x, thresh=SUPPRESS_BETAS_THRESH)
        )
    if SUPPRESS_BAD_POSES:
        dataset = dataset.map(lambda x: supress_bad_poses(x))
    if POSES_BETAS_SIMULTANEOUS:
        dataset = dataset.map(lambda x: poses_betas_simultaneous(x))
    if FILTER_NO_POSES:
        dataset = dataset.select(lambda x: filter_no_poses(x))
    if FILTER_NUM_KP > 0:
        dataset = dataset.select(
            lambda x: filter_numkp(x, numkp=FILTER_NUM_KP, thresh=FILTER_NUM_KP_THRESH)
        )
    if FILTER_REPROJ_THRESH > 0:
        dataset = dataset.select(
            lambda x: filter_reproj_error(x, thresh=FILTER_REPROJ_THRESH)
        )
    if FILTER_MIN_BBOX_SIZE > 0:
        dataset = dataset.select(
            lambda x: filter_bbox_size(x, thresh=FILTER_MIN_BBOX_SIZE)
        )
    if BETAS_REG:
        dataset = dataset.map(
            lambda x: set_betas_for_reg(x)
        )  # NOTE: Must be at the end

    use_skimage_antialias = False
    border_mode = cv2.BORDER_CONSTANT

    # Process the dataset further
    dataset = dataset.map(
        lambda x: process_webdataset_tar_item(
            x,
            train,
            SCALE_FACTOR=SCALE_FACTOR,
            ROT_FACTOR=ROT_FACTOR,
            TRANS_FACTOR=TRANS_FACTOR,
            COLOR_SCALE=COLOR_SCALE,
            ROT_AUG_RATE=ROT_AUG_RATE,
            DO_FLIP=DO_FLIP,
            FLIP_AUG_RATE=FLIP_AUG_RATE,
            EXTREME_CROP_AUG_RATE=EXTREME_CROP_AUG_RATE,
            EXTREME_CROP_AUG_LEVEL=EXTREME_CROP_AUG_LEVEL,
            MEAN=MEAN,
            STD=STD,
            IMG_SIZE=IMG_SIZE,
            BBOX_SHAPE=BBOX_SHAPE,
            use_skimage_antialias=use_skimage_antialias,
            border_mode=border_mode,
        )
    )
    if epoch_size is not None:
        dataset = dataset.with_epoch(epoch_size)

    return dataset


def process_webdataset_tar_item(
    item,
    train,
    SCALE_FACTOR: float = 0.3,
    ROT_FACTOR: int = 30,
    TRANS_FACTOR: float = 0.02,
    COLOR_SCALE: float = 0.2,
    ROT_AUG_RATE: float = 0.6,
    DO_FLIP: bool = True,
    FLIP_AUG_RATE: float = 0.5,
    EXTREME_CROP_AUG_RATE: float = 0.10,
    EXTREME_CROP_AUG_LEVEL: float = 1,
    MEAN=DEFAULT_MEAN,
    STD=DEFAULT_STD,
    IMG_SIZE=DEFAULT_IMG_SIZE,
    BBOX_SHAPE=None,
    use_skimage_antialias=False,
    border_mode=cv2.BORDER_CONSTANT,
):
    do_augmentation = True if train else False

    # Read data from item
    key = item["__key__"]
    image = item["jpg"]
    data = item["data.pyd"]
    mask = item["mask"]

    keypoints_2d = data["keypoints_2d"]
    keypoints_3d = data["keypoints_3d"]
    center = data["center"]
    scale = data["scale"]
    body_pose = data["body_pose"]
    betas = data["betas"]
    has_body_pose = data["has_body_pose"]
    has_betas = data["has_betas"]
    # image_file = data['image_file']

    # Process data
    orig_keypoints_2d = keypoints_2d.copy()
    center_x = center[0]
    center_y = center[1]
    bbox_size = expand_to_aspect_ratio(
        scale * 200, target_aspect_ratio=BBOX_SHAPE
    ).max()
    if bbox_size < 1:
        breakpoint()

    smpl_params = {
        "global_orient": body_pose[:3],
        "body_pose": body_pose[3:],
        "betas": betas,
    }

    has_smpl_params = {
        "global_orient": has_body_pose,
        "body_pose": has_body_pose,
        "betas": has_betas,
    }

    smpl_params_is_axis_angle = {
        "global_orient": True,
        "body_pose": True,
        "betas": False,
    }

    # Crop image and (possibly) perform data augmentation
    img_rgba = np.concatenate([image, mask.astype(np.uint8)[:, :, None] * 255], axis=2)
    (
        img_patch_rgba,
        keypoints_2d,
        keypoints_3d,
        smpl_params,
        has_smpl_params,
        img_size,
        trans,
    ) = get_example(
        img_rgba,
        center_x,
        center_y,
        bbox_size,
        bbox_size,
        keypoints_2d,
        keypoints_3d,
        smpl_params,
        has_smpl_params,
        _FLIP_KEYPOINT_PERMUTATION,
        IMG_SIZE,
        IMG_SIZE,
        MEAN,
        STD,
        do_augmentation,
        SCALE_FACTOR=SCALE_FACTOR,
        ROT_FACTOR=ROT_FACTOR,
        TRANS_FACTOR=TRANS_FACTOR,
        COLOR_SCALE=COLOR_SCALE,
        ROT_AUG_RATE=ROT_AUG_RATE,
        DO_FLIP=DO_FLIP,
        FLIP_AUG_RATE=FLIP_AUG_RATE,
        EXTREME_CROP_AUG_RATE=EXTREME_CROP_AUG_RATE,
        EXTREME_CROP_AUG_LEVEL=EXTREME_CROP_AUG_LEVEL,
        is_bgr=False,
        return_trans=True,
        use_skimage_antialias=use_skimage_antialias,
        border_mode=border_mode,
    )
    img_patch = img_patch_rgba[:3, :, :]
    mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
    if (mask_patch < 0.5).all():
        mask_patch = np.ones_like(mask_patch)

    item = {}

    item["img"] = img_patch
    item["mask"] = mask_patch
    # item['img_og'] = image
    # item['mask_og'] = mask
    item["keypoints_2d"] = keypoints_2d.astype(np.float32)
    item["keypoints_3d"] = keypoints_3d.astype(np.float32)
    item["orig_keypoints_2d"] = orig_keypoints_2d
    item["box_center"] = center.copy()
    item["box_size"] = bbox_size
    item["img_size"] = 1.0 * img_size[::-1].copy()
    item["smpl_params"] = smpl_params
    item["has_smpl_params"] = has_smpl_params
    item["smpl_params_is_axis_angle"] = smpl_params_is_axis_angle
    item["_scale"] = scale
    item["_trans"] = trans
    item["imgname"] = key
    # item['idx'] = idx
    return item
