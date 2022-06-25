import os

import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post
from PIL import Image

# Load color image
file_path = "distorted_image"
output_base = "setelah/"
(xcenter, ycenter, list_fact) = io.load_metadata_txt("dapat/coefficients.txt")


"""function for load all image using for loop os"""
def load_all_image(path):
    list_image = []
    for pathh in os.listdir(path):
        path_ambil = str(path + "/" + pathh)
        # print(pathh)
        mat = np.asarray(Image.open(path_ambil), dtype=np.float32)
        for a in range(mat.shape[-1]):
            mat[:, :, a] = post.unwarp_image_backward(mat[:, :, a], xcenter, ycenter, list_fact)
        # print(str(pathh))
        io.save_image(output_base+ str(pathh), mat)
        print(pathh)
        # list_image.append(mat)
    return list_image

load_all_image(file_path)