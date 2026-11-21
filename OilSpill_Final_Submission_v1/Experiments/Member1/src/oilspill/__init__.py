"""Oil-spill semantic segmentation utilities."""

CLASS_NAMES = ("sea", "oil_spill", "look_alike", "ship", "land")
CLASS_COLORS = (
    (0, 0, 0),
    (0, 255, 255),
    (255, 0, 0),
    (153, 76, 0),
    (0, 153, 0),
)
NUM_CLASSES = len(CLASS_NAMES)
