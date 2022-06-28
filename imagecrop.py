import os

import cv2
from PIL import Image

# Opens a image in RGB mode
from matplotlib import pyplot as plt




def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


for filename in os.listdir("setelah"):
    im = Image.open(r"setelah/" + filename)
    widht, height = im.size
    if widht >= 255 and height >= 255:
        AA = crop_center(im, 255, 255)
    else:
        AA = im
    AA.save("A/" + filename + "_cropped.jpeg")
