import numpy as np

from panel360.augmentation import augment_dataset, augment_thermal_image


def test_augment_thermal_image_preserves_shape_and_values():
    image = np.arange(1024, dtype=np.float32).reshape(32, 32, 1)
    rng = np.random.RandomState(0)

    augmented = augment_thermal_image(image, rng=rng)

    assert augmented.shape == (32, 32, 1)
    # Rotation/flip only rearranges pixels, so the sum (and value set) is preserved.
    np.testing.assert_allclose(np.sort(augmented.ravel()), np.sort(image.ravel()))


def test_augment_dataset_shape():
    images = np.stack([np.full((32, 32, 1), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)])

    augmented = augment_dataset(images, num_augmentations=2)

    assert augmented.shape == (6, 32, 32, 1)
