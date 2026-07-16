"""Light geometric augmentation for 32x32 thermal panel images."""

import cv2
import numpy as np

PANEL_SIZE = 32
FLIP_ROTATE_PROBABILITY = 0.1


def augment_thermal_image(thermal_image, rng=np.random):
    """Randomly rotate/flip a single ``(32, 32, 1)`` thermal image.

    Each transform is applied independently with low probability so most
    images pass through unchanged.
    """
    image = thermal_image.squeeze()
    p = [1 - FLIP_ROTATE_PROBABILITY, FLIP_ROTATE_PROBABILITY]
    if rng.choice([0, 1], p=p):
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rng.choice([0, 1], p=p):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rng.choice([0, 1], p=p):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if rng.choice([0, 1], p=p):
        image = cv2.flip(image, 0)
    elif rng.choice([0, 1], p=p):
        image = cv2.flip(image, 1)
    elif rng.choice([0, 1], p=p):
        image = cv2.flip(image, -1)

    return image.reshape((PANEL_SIZE, PANEL_SIZE, 1))


def augment_dataset(images, num_augmentations=1, rng=np.random):
    """Return ``num_augmentations`` augmented passes over ``images``, stacked."""
    augmented = [augment_thermal_image(img, rng=rng) for _ in range(num_augmentations) for img in images]
    return np.array(augmented)
