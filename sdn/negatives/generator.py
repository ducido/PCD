"""Counterfactual (negative) observations for SDN: remove task-relevant objects from the image.

Task-relevant objects are parsed from the instruction, segmented (Grounding-DINO + SAM2
tracking by default, or the simulator ground truth with by='gt') and removed by
  - 'zeros_bbox'        : zeroing their padded bounding box (default in the paper),
  - 'inpaint'           : LaMa inpainting,
  - 'random_zeros_bbox' : zeroing an equally sized box elsewhere (control, Table V).
"""

import cv2
import numpy as np
import torch

from .instruction_templates import get_objects_from_instruction
from .mask_predictors import build_predictor, predict_masks_with_predictor
from .properties import _ROBOT_NAMES
from .utils import dilate_mask

NEGATIVE_MODES = ['zeros_bbox', 'inpaint', 'random_zeros_bbox']


def mask_with_bbox_zero(rgb_image, mask, pad=3):
    """Zero the padded bounding box of `mask`; returns the image unchanged if the mask is empty."""
    masked_image = rgb_image.copy()
    if mask is None:
        return masked_image

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return masked_image

    H, W = mask.shape
    x_min, x_max = max(0, xs.min() - pad), min(W - 1, xs.max() + pad)
    y_min, y_max = max(0, ys.min() - pad), min(H - 1, ys.max() + pad)
    masked_image[y_min:y_max + 1, x_min:x_max + 1] = 0
    return masked_image


def _free_box_positions(integral, box_h, box_w):
    """All (top, left) where a box_h x box_w window contains 0 forbidden pixels."""
    box_sum = (integral[box_h:, box_w:] - integral[:-box_h, box_w:]
               - integral[box_h:, :-box_w] + integral[:-box_h, :-box_w])
    return np.argwhere(box_sum == 0)


def mask_with_random_bbox_zero(rgb_image, mask, excluded_mask=None, pad=3, margin=10,
                               min_size=4, shrink=0.9):
    """
    Zero out a bbox at a random position that does not overlap mask / excluded_mask
    and keeps at least `margin` pixels of distance from them.

    The box starts at the size mask_with_bbox_zero would use (padded bbox of mask)
    and is shrunk by `shrink` until a valid position exists, down to `min_size`.
    """
    masked_image = rgb_image.copy()
    if mask is None:
        return masked_image

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return masked_image

    H, W = mask.shape

    # initial box size = padded bbox of mask (same as mask_with_bbox_zero)
    x_min, x_max = max(0, xs.min() - pad), min(W - 1, xs.max() + pad)
    y_min, y_max = max(0, ys.min() - pad), min(H - 1, ys.max() + pad)
    box_h = min(H, y_max - y_min + 1)
    box_w = min(W, x_max - x_min + 1)

    # forbidden region: mask + excluded_mask, grown by `margin` for the distance constraint
    forbidden = mask.astype(bool).copy()
    if excluded_mask is not None:
        forbidden |= excluded_mask.astype(bool)
    forbidden = dilate_mask(forbidden, 2 * margin + 1)
    integral = cv2.integral(forbidden.astype(np.uint8), sdepth=cv2.CV_32S)

    # shrink the box (keeping aspect ratio) until it fits somewhere
    cur_h, cur_w = box_h, box_w
    while True:
        candidates = _free_box_positions(integral, cur_h, cur_w)
        if len(candidates) > 0:
            break
        if cur_h <= min_size and cur_w <= min_size:
            return masked_image
        cur_h = max(min_size, int(cur_h * shrink))
        cur_w = max(min_size, int(cur_w * shrink))

    top, left = candidates[np.random.randint(len(candidates))]
    masked_image[top:top + cur_h, left:left + cur_w] = 0
    return masked_image


def name_to_alias(name):
    rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
    # also drop object sizes such as "10cm"
    return ' '.join(w for w in name.split('_') if w[-2:] != "cm" and w not in rm_list)


class NegativeImageGenerator:
    """Builds the perturbed observation o_neg used to sample the negative set B."""

    def __init__(self,
                 env,
                 camera_name=None,
                 by='grounded_sam_tracking',
                 negative_mode='zeros_bbox',
                 inpaint_mode='lama',
                 bbox_pad=3,
                 random_bbox_margin=10,
                 version=2,
                 get_all_parts=False):
        if negative_mode not in NEGATIVE_MODES:
            raise ValueError(f"negative_mode must be one of {NEGATIVE_MODES}, got {negative_mode}")
        assert version in [2, 3]
        self.env = env
        self.camera_name = camera_name
        self.by = by
        self.negative_mode = negative_mode
        self.bbox_pad = bbox_pad
        self.random_bbox_margin = random_bbox_margin
        self.version = version
        self.get_all_parts = get_all_parts

        self.mask_objects = None
        self.keep_objects = None
        self.task_description = None
        self.predictor = build_predictor(by) if by != 'gt' else None
        self.inpainter = None
        if negative_mode == 'inpaint':
            from .inpainters import build_inpainter
            self.inpainter = build_inpainter(inpaint_mode)

    def generate(self, obs, task_description):
        if task_description != self.task_description:
            self._start_new_instruction(obs, task_description)

        if self.by == 'gt':
            mask, excluded_mask = self.get_mask_by_gt(obs)
        else:
            mask, excluded_mask = self.get_mask_by_predictor(obs)

        rgb_image = self._get_rgb_image(obs)
        if self.negative_mode == 'inpaint':
            return self.inpainter.inpaint(rgb_image, mask, excluded_mask)
        if self.negative_mode == 'zeros_bbox':
            return mask_with_bbox_zero(rgb_image, mask, pad=self.bbox_pad)
        return mask_with_random_bbox_zero(rgb_image, mask, excluded_mask,
                                          pad=self.bbox_pad, margin=self.random_bbox_margin)

    def reset(self):
        self.task_description = None

    def _start_new_instruction(self, obs, task_description):
        self.task_description = task_description
        self.mask_objects = get_objects_from_instruction(task_description, self.get_all_parts)
        self.keep_objects = _ROBOT_NAMES if self.by == 'gt' else ['robot']
        if self.predictor is not None:
            # restart detection / tracking for the new instruction
            if hasattr(self.predictor, 'reset'):
                self.predictor.reset()
            if self.by in ['point_tracking', 'box_tracking']:
                self._set_points_or_boxes(obs)

    def get_mask_by_gt(self, obs):
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()

        masks = [self._get_object_mask_by_gt(seg, name2id, name) for name in self.mask_objects]
        keep_masks = [self._get_object_mask_by_gt(seg, name2id, name) for name in self.keep_objects]
        mask = self._merge_masks(seg.shape, masks, keep_masks)

        robot_mask = np.zeros_like(seg, dtype=bool)
        for robot_name in _ROBOT_NAMES:
            robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)
        return mask, robot_mask

    def get_mask_by_predictor(self, obs):
        image = self._get_rgb_image(obs)
        objs = self.mask_objects + self.keep_objects
        masks = predict_masks_with_predictor(image, objs, self.predictor)
        num_mask = len(self.mask_objects)
        mask_obj_masks, keep_obj_masks = masks[:num_mask], masks[num_mask:num_mask + len(self.keep_objects)]
        robot_mask = masks[objs.index('robot')] if 'robot' in objs else None
        mask = self._merge_masks(image.shape[:2], mask_obj_masks, keep_obj_masks)
        return mask, robot_mask

    def _set_points_or_boxes(self, obs):
        """Oracle visual prompts from the simulator segmentation (by='point_tracking'/'box_tracking')."""
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()

        masks = []
        for obj_name in self.mask_objects + self.keep_objects:
            if obj_name == 'robot':
                robot_mask = np.zeros_like(seg, dtype=bool)
                for robot_name in _ROBOT_NAMES:
                    robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)
                masks.append(robot_mask)
            else:
                masks.append(self._get_object_mask_by_gt(seg, name2id, obj_name))

        if self.by == 'point_tracking':
            self.predictor.predictor.set_points([_mask_to_points(mask) for mask in masks])
        else:
            self.predictor.predictor.set_boxes([_mask_to_bbox(mask) for mask in masks])

    def _get_name_to_id(self):
        if self.version == 2:
            actor_name2id = {name_to_alias(actor.name): actor.id for actor in self.env.unwrapped.get_actors()}
            robot_name2id = {link.name: link.id for link in self.env.unwrapped.agent.robot.get_links()}
            art_name2id = {}
            for art_obj in self.env.unwrapped.get_articulations():
                if art_obj.name in ['cabinet']:
                    for link in art_obj.get_links():
                        art_name2id[name_to_alias(link.name)] = link.id
            return {**actor_name2id, **robot_name2id, **art_name2id}
        return {v.name: k for k, v in self.env.unwrapped.segmentation_id_map.items()}

    @staticmethod
    def _get_object_mask_by_gt(seg, name2id, obj_name):
        if obj_name not in name2id:
            return np.zeros_like(seg, dtype=bool)
        return seg == name2id[obj_name]

    @staticmethod
    def _merge_masks(shape, masks, keep_masks):
        mask = np.zeros(shape, dtype=bool)
        for obj_mask in masks:
            if obj_mask is not None:
                mask |= obj_mask
        for obj_mask in keep_masks:
            if obj_mask is not None:
                mask[obj_mask] = False
        return mask

    def _get_rgb_image(self, obs):
        image = self._get_camera_images(obs)['rgb']
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0]
        return image

    def _get_segmentation(self, obs):
        if self.version == 2:
            seg = self._get_camera_images(obs)["Segmentation"][..., 1].copy()
        else:
            seg = self._get_camera_images(obs)["segmentation"][0, :, :, 0]
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        return seg

    def _get_camera_images(self, obs):
        robot = self.env.unwrapped.robot_uid if self.version == 2 else self.env.unwrapped.robot_uids
        if not isinstance(robot, list):
            robot = ''.join(robot)

        camera_name = self.camera_name
        if camera_name is None:
            if "google_robot" in robot:
                camera_name = "overhead_camera"
            elif "widowx" in robot:
                camera_name = "3rd_view_camera"
            elif "panda" in robot:
                camera_name = "base_camera"
            else:
                raise NotImplementedError(f"No default camera for robot {robot}")

        key = "image" if self.version == 2 else "sensor_data"
        return obs[key][camera_name]


def _mask_to_points(mask, num_points=5):
    if not mask.any():
        return None
    points = np.argwhere(mask)
    if len(points) > num_points:
        points = points[np.random.choice(len(points), num_points, replace=False)]
    return points[:, [1, 0]]  # (y, x) -> (x, y)


def _mask_to_bbox(mask):
    if not mask.any():
        return None
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])
