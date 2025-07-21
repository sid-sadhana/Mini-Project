import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import random

def apply_image_augmentation(img, method):
    if method == "Rotate":
        return img.rotate(random.uniform(-30, 30))

    elif method == "Crop":
        w, h = img.size
        return img.crop((w * 0.1, h * 0.1, w * 0.9, h * 0.9)).resize((w, h))

    elif method == "Flip Horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    elif method == "Flip Vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    elif method == "Gaussian Noise":
        np_img = np.array(img)
        noise = np.random.normal(0, 15, np_img.shape).astype(np.uint8)
        noised = np.clip(np_img + noise, 0, 255)
        return Image.fromarray(noised.astype(np.uint8))

    elif method == "Increase Brightness":
        return ImageEnhance.Brightness(img).enhance(1.5)

    elif method == "Decrease Brightness":
        return ImageEnhance.Brightness(img).enhance(0.5)

    elif method == "Increase Contrast":
        return ImageEnhance.Contrast(img).enhance(1.5)

    elif method == "Blur":
        return img.filter(ImageFilter.GaussianBlur(2))

    elif method == "Color Jitter":
        transform = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        return transform(img)

    elif method == "Zoom":
        w, h = img.size
        return img.crop((w*0.1, h*0.1, w*0.9, h*0.9)).resize((w, h))

    elif method == "Shear":
        transform = T.RandomAffine(degrees=0, shear=15)
        return transform(img)
