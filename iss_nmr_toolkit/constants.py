"""Shared constants for the offline ISS/NMR toolkit."""

    # DEFAULT_VIEWS = {
    #     "front": "exterior_image_1_left",
    #     "wrist": "wrist_image_left",
    #     "overhead": "overhead_image",
    # }

DEFAULT_VIEWS = {
    "front": "image",
    # "wrist": "wrist_image",
}

VIEW_NAMES = tuple(DEFAULT_VIEWS.keys())

# LeRobot video keys holding the object-of-interest segmentation per view.
MASK_VIDEO_KEYS = {
    "front": "observation.images.object_of_interest_mask",
    "wrist": "observation.images.object_of_interest_wrist_mask",
}

LABEL_NUIS = 0
LABEL_SUP = 1
LABEL_ACT = 2

