import cv2
import numpy as np
import os
import torch

from .inpainters import build_inpainter
from .instruction_templates import get_objects_from_instruction
from .mask_predictors import build_predictor, predict_masks_with_predictor
from .properties import _ROBOT_NAMES
from .utils import dilate_mask, visualize_multi_objects


def mask_with_bbox_noise(rbg_image, mask, pad=10):
    """
    rbg_image: (H, W, 3)
    mask: (H, W) binary (0/1 hoặc bool)
    pad: số pixel mở rộng bbox
    """

    masked_image = rbg_image.copy()

    ys, xs = np.where(mask > 0)

    # nếu không có object thì return ảnh gốc
    if len(xs) == 0 or len(ys) == 0:
        return masked_image

    # bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # padding
    H, W = mask.shape
    x_min = max(0, x_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(H - 1, y_max + pad)

    # tạo noise
    noise = np.random.randint(
        0, 256,
        size=(y_max - y_min + 1, x_max - x_min + 1, 3),
        dtype=np.uint8
    )

    # fill rectangle bằng noise
    masked_image[y_min:y_max+1, x_min:x_max+1] = noise

    return masked_image

def mask_with_bbox_zero(rbg_image, mask, pad=10):
    if mask is None:
        # không có mask → return ảnh gốc
        return rbg_image

    masked_image = rbg_image.copy()

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return masked_image

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    H, W = mask.shape
    x_min = max(0, x_min - pad)
    x_max = min(W - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(H - 1, y_max + pad)

    masked_image[y_min:y_max+1, x_min:x_max+1] = 0

    return masked_image

def mask_to_points(mask):
    if not mask.any():
        return None
    points = np.argwhere(mask)
    # sample 5 points
    if len(points) > 5:
        points = points[np.random.choice(len(points), 5, replace=False)]
    # y,x -> x,y
    return points[:, [1, 0]]


def mask_to_bbox(mask):
    if not mask.any():
        return None
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])
    

def _free_box_positions(integral, box_h, box_w):
    """All (top, left) where a box_h x box_w window contains 0 forbidden pixels."""
    box_sum = (integral[box_h:, box_w:] - integral[:-box_h, box_w:]
               - integral[box_h:, :-box_w] + integral[:-box_h, :-box_w])
    return np.argwhere(box_sum == 0)


def mask_with_random_bbox_zero(rbg_image, mask, excluded_mask=None, pad=3, margin=10,
                               min_size=4, shrink=0.9):
    """
    Zero out a bbox at a random position that does not overlap mask / excluded_mask
    and keeps at least `margin` pixels of distance from them.

    The box starts at the size mask_with_bbox_zero would use (padded bbox of mask)
    and is shrunk by `shrink` until a valid position exists, down to `min_size`.
    """
    if mask is None:
        return rbg_image

    masked_image = rbg_image.copy()

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return masked_image

    H, W = mask.shape

    # initial box size = padded bbox of mask (same as mask_with_bbox_zero)
    x_min = max(0, xs.min() - pad)
    x_max = min(W - 1, xs.max() + pad)
    y_min = max(0, ys.min() - pad)
    y_max = min(H - 1, ys.max() + pad)
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
            print(f"[random bbox] no free {min_size}x{min_size} spot (margin={margin}), "
                  "returning image unchanged")
            return masked_image
        cur_h = max(min_size, int(cur_h * shrink))
        cur_w = max(min_size, int(cur_w * shrink))

    if (cur_h, cur_w) != (box_h, box_w):
        print(f"[random bbox] shrunk box {box_h}x{box_w} -> {cur_h}x{cur_w} to fit")

    top, left = candidates[np.random.randint(len(candidates))]
    masked_image[top:top + cur_h, left:left + cur_w] = 0

    return masked_image

def name_to_alias(name):
    s = name.split('_')
    rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
    cleaned = []
    for w in s:
        if w[-2:] == "cm":
            # object size in object name
            continue
        if w not in rm_list:
            cleaned.append(w)
    return ' '.join(cleaned)


def my_print(*args):
    # get gpu id
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    if gpu_id == '0':
        print(args)


class ContrastImageGenerator:
    def __init__(self, 
                 env, 
                 camera_name=None, 
                 by="gt",
                 inpaint_mode="lama",
                 color="auto",
                 sigma=5,
                 version=2,
                 get_all_parts=False):
        self.env = env
        self.camera_name = camera_name
        self.by = by
        self.color = color
        self.sigma = sigma
        assert version in [2, 3]
        self.version = version
        self.get_all_parts = get_all_parts
        
        self.mask_objects = None
        self.keep_objects = None
        self.task_description = None
        self.predictor = None
        self.inpainter = build_inpainter(inpaint_mode)
    
    def generate(self, obs, task_description, logging=None, is_inpaint=True):
        if task_description != self.task_description:
            self.reset_mask_and_keep_object_names(task_description)
            self.task_description = task_description
            if self.by != "gt":
                self.predictor = build_predictor(self.by)
                if self.by in ["point_tracking", "box_tracking"]:
                    self._set_points_or_boxes(obs)
        
        if self.by == "gt":
            mask, excluded_mask = self.get_mask_by_gt(obs, reverse_mask=False)
        else:
            mask, excluded_mask = self.get_mask_by_predictor(obs, reverse_mask=False)

        if is_inpaint:
            image = self.inpainter.inpaint(self._get_rgb_image(obs), mask, excluded_mask)
        else:

            # logging.info("No inpainting, masking objects keep shape")
            # rbg_image = self._get_rgb_image(obs)
            # masked_image = np.where(mask[..., None] == 0, rbg_image, 0)

            # logging.info("No inpainting, masking objects with bbox noise")
            # rbg_image = self._get_rgb_image(obs)
            # masked_image = mask_with_bbox_noise(rbg_image, mask, pad=10)
            # image = masked_image

            logging.info("No inpainting, masking objects with bbox zero pad 3")
            rbg_image = self._get_rgb_image(obs)
            masked_image = mask_with_bbox_zero(rbg_image, mask, pad=3)
            # logging.info("No inpainting, masking GRIPPER with bbox zero pad 15")
            # masked_image = mask_with_bbox_zero(rbg_image, excluded_mask, pad=15)

            # print("No inpainting, zeroing a random bbox away from mask/excluded_mask")
            # rbg_image = self._get_rgb_image(obs)
            # masked_image = mask_with_random_bbox_zero(rbg_image, mask, excluded_mask, pad=3, margin=10)


            image = masked_image
        return image
    
    def reset(self):
        self.task_description = None
        
    def reset_mask_and_keep_object_names(self, task_description):
        self.mask_objects = get_objects_from_instruction(task_description, self.get_all_parts)
        self.keep_objects = _ROBOT_NAMES if self.by == "gt" else ["robot"]
        # my_print('reset:', self.mask_objects, self.keep_objects)
    
    def get_mask_by_gt(self, obs, reverse_mask=False):
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()
        
        masks = [self._get_object_mask_by_gt(seg, name2id, obj_name) for obj_name in self.mask_objects]
        keep_masks = [self._get_object_mask_by_gt(seg, name2id, obj_name) for obj_name in self.keep_objects]
        mask = self._add_reserve_keep_mask(seg.shape, masks, reverse_mask, keep_masks)

        robot_mask = np.zeros_like(seg, dtype=bool)
        for robot_name in _ROBOT_NAMES:
            robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)
            
        # visualize_multi_objects(self._get_rgb_image(obs), masks + [robot_mask], self.mask_objects + ['robot'], 'test.jpg') 
        return mask, robot_mask
    
    def get_mask_by_predictor(self, obs, reverse_mask=False):
        image = self._get_rgb_image(obs)
        objs = self.mask_objects + self.keep_objects
        masks = predict_masks_with_predictor(image, objs, self.predictor)
        # my_print('get:', len(masks), objs)
        mask_obj_masks, keep_obj_masks = masks[:len(self.mask_objects)], masks[len(self.mask_objects):len(self.mask_objects) + len(self.keep_objects)]
        robot_mask = masks[objs.index('robot')] if 'robot' in objs else None
        mask = self._add_reserve_keep_mask(image.shape[:2], mask_obj_masks, reverse_mask, keep_obj_masks)
        return mask, robot_mask
    
    def _set_points_or_boxes(self, obs):
        assert self.by in ["point_tracking", "box_tracking"]
        assert self.predictor is not None, "Predictor is not initialized"
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

        if self.by == "point_tracking":
            points = [mask_to_points(mask) for mask in masks]
            self.predictor.predictor.set_points(points)
        elif self.by == "box_tracking":
            boxes = [mask_to_bbox(mask) for mask in masks]
            self.predictor.predictor.set_boxes(boxes)
    
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
        else:
            name2id = {}
            for k, v in self.env.unwrapped.segmentation_id_map.items():
                name2id[v.name] = k
            return name2id

    def _get_object_mask_by_gt(self, seg, name2id, obj_name):
        # 1. check if object name is in assets
        if os.path.exists('assets'):
            filenames = os.listdir('assets')
            if obj_name + '.png' in filenames:
                return cv2.imread(f'assets/{obj_name}.png', cv2.IMREAD_GRAYSCALE) > 0
        
        # 2. check if object name is in name2id
        if obj_name not in name2id:
            return np.zeros_like(seg, dtype=bool)
        return seg == name2id[obj_name]
    
    def _add_reserve_keep_mask(self, shape, masks, reverse_mask, keep_masks):
        mask = np.zeros(shape, dtype=bool)
        for obj_mask in masks:
            if obj_mask is not None:
                mask |= obj_mask

        if reverse_mask:
            mask = ~mask

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
        seg_key = "Segmentation" if self.version == 2 else "segmentation"
        if self.version == 2:
            seg = self._get_camera_images(obs)[seg_key][..., 1].copy()
        else:
            seg = self._get_camera_images(obs)[seg_key][0, :, : ,0]
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
            elif "panda_wristcam" in robot:
                camera_name = "base_camera"
            else:
                raise NotImplementedError()
        
        key = "image" if self.version == 2 else "sensor_data"
        return obs[key][camera_name]
