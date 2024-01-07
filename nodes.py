import sys
import os
from PIL import Image, ImageOps
import folder_paths
import torch
import numpy as np
import requests
from io import BytesIO


def open_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    return image


def back_image(i):
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))


class LoadImageFromUrl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, url):
        if url != "" and url.startswith('http'):
            i = open_image_from_url(url)
            return back_image(i)
        return None, None


NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrl": LoadImageFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromUrl": "Load Image From URL",
}
