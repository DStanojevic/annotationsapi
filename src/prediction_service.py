from typing import Optional
import numpy as np
import cv2
from segment_anything import SamPredictor
from dtos import Square, Point


def predict_annotation(predictor: SamPredictor, box: Square) -> Optional[np.ndarray]:
    input_box = _get_box(box)
    masks, scores, logits = predictor.predict(box=input_box, multimask_output=True)
    index = np.argmax(scores)
    mask = masks[index]
    return _binary_mask_to_polyline(mask)


def _get_box(box: Square) -> np.ndarray:
    return np.array([box.topLeft.x, box.topLeft.y, box.bottomRight.x, box.bottomRight.y])


def _binary_mask_to_polyline(binary_mask: np.ndarray):
    # find the contours in the binary mask
    converted_mask = (binary_mask * 255).astype(np.uint8)
    contours, t1 = cv2.findContours(converted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polylines = []
    for contour in contours:
        # convert contour to polyline (list of points)
        polyline = [Point(x=point[0][0], y=point[0][1]) for point in contour]
        polylines.append(polyline)

    return polylines
