import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.post.postprocessing as post

# Load image
mat0 = io.load_image("distorted_image/0_0.jpeg")
output_base = "dapat/"
(height, width) = mat0.shape
mat0 = mat0 / np.max(mat0)

# Estimated forward model
xcenter = width / 2.0 + 110.0
ycenter = height / 2.0 - 20.0
list_pow = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
list_coef = np.asarray([1.0, 4.0, 5.0, 17.0, 3.0])
list_ffact = list_pow * list_coef

# Calculate parameters of a backward model from the estimated forward model
list_hor_lines = []
for i in range(20, height-20, 50):
    list_tmp = []
    for j in range(20, width-20, 50):
        list_tmp.append([i - ycenter, j - xcenter])
    list_hor_lines.append(list_tmp)
Amatrix = []
Bmatrix = []
list_expo = np.arange(len(list_ffact), dtype=np.int16)
for _, line in enumerate(list_hor_lines):
    for _, point in enumerate(line):
        xd = np.float64(point[1])
        yd = np.float64(point[0])
        rd = np.sqrt(xd * xd + yd * yd)
        ffactor = np.float64(np.sum(list_ffact * np.power(rd, list_expo)))
        if ffactor != 0.0:
            Fb = 1 / ffactor
            ru = ffactor * rd
            Amatrix.append(np.power(ru, list_expo))
            Bmatrix.append(Fb)
Amatrix = np.asarray(Amatrix, dtype=np.float64)
Bmatrix = np.asarray(Bmatrix, dtype=np.float64)
list_bfact = np.linalg.lstsq(Amatrix, Bmatrix, rcond=1e-64)[0]

# Apply distortion correction
corrected_mat = post.unwarp_image_backward(mat0, xcenter, ycenter, list_bfact)
io.save_image(output_base + "/sebelum.jpeg", corrected_mat)
io.save_image(output_base + "/sesudah.jpeg", mat0)
io.save_metadata_txt(output_base + "/coefficients.txt", xcenter, ycenter, list_bfact)