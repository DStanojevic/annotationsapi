from typing import Optional
import cv2

_image_locations = {
    'img1': '../assets/images/10x_HeLa_Kyoto.jpg'
}

_images = {}


def get_image(image_id: str) -> Optional[cv2.UMat]:
    images = _get_images()
    return images.get(image_id, _load_image(image_id))


def _load_image(image_id: str) -> Optional[cv2.UMat]:
    image_locations = _get_image_locations()
    image_location = image_locations.get(image_id)
    if image_location:
        image_bgr = cv2.imread(image_location)
        image_content = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return _add_image(image_id, image_location, image_content)[image_id]


def _add_image(image_id: str, image_location: str, image_content: cv2.UMat) -> dict:
    images = _get_images()
    images[image_id] = {
        'location': image_location,
        'content': image_content
    }
    return images


def _get_image_locations():
    global _image_locations
    return _image_locations


def _get_images():
    global _images
    return _images
