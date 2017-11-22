import numpy as np


def get_patches(img, max_patches, patch_size):
    h = img.shape[0]
    w = img.shape[1]
    n = 0
    while n < max_patches:
        start_r = np.random.randint(0, h - patch_size, 1)[0]
        start_c = np.random.randint(0, w - patch_size, 1)[0]
        patch = img[start_r:start_r + patch_size, start_c:start_c + patch_size, :]
        n = n + 1
        yield patch
