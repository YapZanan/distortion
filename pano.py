import os

import stitching
from matplotlib import pyplot as plt
stitcher = stitching.Image_Stitching()
ima = []
for filename in os.listdir("B"):
    im = ("A/" + filename)
    ima.append(im)

panorama = stitcher.min_match([ima])
panorama.show()