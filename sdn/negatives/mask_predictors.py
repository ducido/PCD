import cv2
import numpy as np
import os
import torch
from PIL import Image

from .utils import *


_CLASSNAMES = ['robot', 'coke can', 'pepsi can', 'redbull can', '7up can', 'blue plastic bottle', 
               'apple', 'orange', 'sponge', 'bottom drawer', 'middle drawer', 'top drawer',
               'eggplant', 'spoon', 'carrot', 'plate larger', 'table cloth shorter', 
               'yellow basket', 'green cube', 'yellow cube', 'goal_site']


_NAME_TO_ALIAS_GDINO = {
    '7up can': 'white 7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    'sponge': 'green sponge',
    'plate larger': 'plate',
    'table cloth shorter': 'blue towel cloth',
    'top drawer': 'dresser',
    'middle drawer': 'dresser',
    'bottom drawer': 'dresser',
    'robot': 'robot manipulator',
}

_SAM2_MODEL_CFG = os.path.join('configs', 'sam2.1', 'sam2.1_hiera_l.yaml')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SAM2_CHECKPOINT = os.path.join(_REPO_ROOT, 'pretrained', 'sam2.1_hiera_large.pt')

_GROUNDING_DINO_CHECKPOINT = os.path.join(_REPO_ROOT, 'pretrained', 'grounding-dino-base')


def postprocess_mask(mask):
    if mask is None or not mask.any():
        return None
    
    # only keep the largest connected component
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    if num_labels > 1:
        # for each component, if area is smaller than 0.1 * largest_area, set it to 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 0.1 * largest_area:
                mask[labels == i] = 0

    # get counter of mask and fill poly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, contours, 1)
    return (mask > 0)


class GroundedSAMPredictor:
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_CHECKPOINT)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GROUNDING_DINO_CHECKPOINT).to('cuda:0')
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image_pil = Image.fromarray(image.copy())
        
        all_prompts = prompts
        text = '. '.join([_NAME_TO_ALIAS_GDINO.get(p, p) for p in all_prompts]) + '.'
        
        inputs = self.processor(image_pil, text=text, return_tensors="pt").to('cuda:0')
        outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        input_boxes = result['boxes'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        labels = []

        # avoid one alias to multi name mapping
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS_GDINO.items() if k in prompts}

        for l in result['labels']:
            # label will remove word "can"
            if l.endswith('pop'):
                l += ' can'
            labels.append(alias_to_name.get(l, l))

        self.image_predictor.set_image(image.copy())
        masks, mask_scores, mask_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        output_masks = []
        for name in prompts:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)

        return [postprocess_mask(mask) for mask in output_masks]


class VisualPromptPredictor:
    def __init__(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.points = None
        self.boxes = None
    
    def set_points(self, points):
        # points is a list contains None
        self.points = np.array([point for point in points if point is not None])
        self.mask_index = [point is not None for point in points]
        
    def set_boxes(self, boxes):
        self.boxes = np.array([box for box in boxes if box is not None])
        self.mask_index = [box is not None for box in boxes]
    
    def predict(self, image, prompts):
        # only support one of points or boxes
        assert self.points is not None or self.boxes is not None, 'points or boxes must be provided!'
        assert self.points is None or self.boxes is None, 'only one of points or boxes can be provided!'
        
        self.image_predictor.set_image(image.copy())
        if self.points is not None:
            point_labels = np.ones(self.points.shape[:-1], dtype=int)
            masks, _, _ = self.image_predictor.predict(point_coords=self.points, point_labels=point_labels, 
                                                       box=None, multimask_output=False)
        elif self.boxes is not None:
            masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None, 
                                                       box=self.boxes, multimask_output=False)
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        out_masks = []
        count = 0
        for mask_index in self.mask_index:
            if mask_index:
                out_masks.append(masks[count])
                count += 1
            else:
                out_masks.append(None)
        return [postprocess_mask(mask) for mask in out_masks]


class TrackingPredictorV2:
    def __init__(self, predictor):
        from sam2.build_sam import build_sam2_camera_predictor
        self.predictor = predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.start_tracking = False
        
    def predict(self, image, prompts):
        if not self.start_tracking:
            masks = self.predictor.predict(image, prompts)
            if all(mask is None for mask in masks):
                return masks
            
            self.video_predictor.load_first_frame(image)
            for mask, prompt in zip(masks, prompts):
                obj_id = _CLASSNAMES.index(prompt) + 1
                if mask is not None:
                    self.video_predictor.add_new_mask(0, obj_id, mask)
                    
            self.start_tracking = True
            return masks
        
        obj_ids, mask_logits = self.video_predictor.track(image)
        name2mask = dict()
        for idx, obj_id in enumerate(obj_ids):
            classname = _CLASSNAMES[obj_id - 1]
            mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
            if not mask.any():
                mask = None
            name2mask[classname] = mask
        
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        from sam2.build_sam import build_sam2_camera_predictor
        self.video_predictor.to('cpu')
        del self.video_predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.start_tracking = False


def build_predictor(predictor_name):
    """
    grounded_sam_tracking : Grounding-DINO detection on the first frame, SAM2 tracking afterwards (paper).
    grounded_sam          : Grounding-DINO + SAM2 on every frame.
    point_tracking / box_tracking : oracle point / box prompts from the simulator, SAM2 tracking.
    """
    if predictor_name == 'grounded_sam':
        return GroundedSAMPredictor()
    if predictor_name == 'grounded_sam_tracking':
        return TrackingPredictorV2(GroundedSAMPredictor())
    if predictor_name in ['point_tracking', 'box_tracking']:
        return TrackingPredictorV2(VisualPromptPredictor())
    raise ValueError(f'predictor_name {predictor_name} is not supported')


def predict_masks_with_predictor(image, prompts, predictor):
    masks = predictor.predict(image, prompts)
    return masks
