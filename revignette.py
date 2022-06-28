import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy import asarray
from numpy import save
from numpy import load
import cv2
import colorcorrect.algorithm as cca

def log_entropy(image, sigma_smooth=2.25):
  N = 256
  n_hist = np.zeros(N, dtype=float)

  rescaled_image = (N - 1) * (image / image.max())
  li = (N - 1) * np.log2(1 + rescaled_image) / np.log2(N)
  li_floor = np.floor(li).astype(int)
  li_ceil = np.ceil(li).astype(int)

  np.add.at(n_hist, li_floor, 1 + li_floor - li)
  np.add.at(n_hist, li_ceil, li_ceil - li)

  n_hist_smooth = gaussian_filter(n_hist, sigma_smooth)

  pk = n_hist_smooth / np.sum(n_hist_smooth)
  log_pk = np.where(pk != 0, np.log2(pk), 0)

  return - np.sum(pk * log_pk)


def compute_center_of_mass(image):
  (sum, i_sum, j_sum) = (0, 0, 0)

  for j in range(image.shape[0]):
    for i in range(image.shape[1]):
      sum += image[j, i]

      i_sum += (i + 1) * image[j, i]
      j_sum += (j + 1) * image[j, i]

  return (i_sum / sum, j_sum / sum)

def compute_r_matrix(image):
  (i_mid, j_mid) = compute_center_of_mass(image)

  res = np.empty_like(image, dtype=float)

  d = np.sqrt(i_mid**2 + j_mid**2)

  for j in range(image.shape[0]):
    for i in range(image.shape[1]):
      res[j, i] = np.sqrt((i - i_mid)**2 + (j - j_mid)**2) / d

  return res

def g(r, a, b, c):
  return 1 + a * r**2 + b * r**4 + c * r**6

def verify_constraints(a, b, c):
  if a > 0 and b == 0 and c == 0:
      return True
  if a >= 0 and b > 0 and c == 0:
      return True
  if c == 0 and b < 0 and -a <= 2 * b:
      return True
  if c > 0 and b**2 < 3 * a * c:
      return True
  if c > 0 and b**2 == 3 * a * c and b >= 0:
      return True
  if c > 0 and b**2 == 3 * a * c and -b >= 3 * c:
      return True
  if c == 0:
      return False
  q_p = (-2 * b + np.sqrt(4 * b**2 - 12 * a * c)) / (6 * c)
  if c > 0 and b**2 > 3 * a * c and q_p <= 0:
      return True
  q_d = (-2 * b - np.sqrt(4 * b**2 - 12 * a * c)) / (6 * c)
  if c > 0 and b**2 > 3 * a * c and q_d >= 1:
      return True
  if c < 0 and b**2 > 3 * a * c and q_p >= 1 and q_d <= 0:
      return True

  return False

def rgb_to_luminance(image):
  return 0.2126 * image[...,0] + 0.7152 * image[...,1] + 0.0722 * image[...,2]

def correct_vignetting(image):
  """
  Apply vignetting correction

  Parameters
  ----------
  image : np.arrray
    The input image to be corrected

  Returns
  -------
  np.array
    The corrected image (array like `image`)
  """
  gray_image = rgb_to_luminance(image)

  (a, b, c) = (0.0, 0.0, 0.0)
  delta = 8.0
  h_min = log_entropy(gray_image)
  r = compute_r_matrix(gray_image)

  while delta > 1 / 256:
    v_arr = np.array([(a + delta, b, c), (a - delta, b, c),
                      (a, b + delta, c), (a, b - delta, c),
                      (a, b, c + delta), (a, b, c - delta)])

    for v in v_arr:
      if verify_constraints(*v):
        h_tmp = log_entropy(gray_image * g(r, *v))
        if h_tmp < h_min:
          h_min = h_tmp
          (a, b, c) = v
          delta = 16.0

    delta /= 2.0

  print(f"Coefficients: ({a}, {b}, {c}), Minimal entropy: {h_min}")
  aa = g(r, a, b, c)
  plt.imshow(aa)
  print(type(aa))
  save('data.npy', aa)

  plt.show()

  res = image * np.stack(3 * [aa], axis=2)

  return np.clip(res, 0, 255).astype(np.uint8)


def pakai_data(image):
  """
  Apply vignetting correction

  Parameters
  ----------
  image : np.arrray
    The input image to be corrected

  Returns
  -------
  np.array
    The corrected image (array like `image`)
  """

  aa = load('data.npy')



  res = image * np.stack(3 * [aa], axis=2)

  return np.clip(res, 0, 255).astype(np.uint8)

# image = plt.imread('A/0_3.jpeg')
# # corrected_image = correct_vignetting(image)
# corrected_image = pakai_data(image)
# plt.imsave('img/sample-3_b-corrected.jpeg', corrected_image)
# plt.imshow(corrected_image)
# plt.show()

# print(np.all(plt.imread('img/sample-3_b-corrected.jpeg') == plt.imread('A/0_3.jpeg')))

# def white_balance(img):
#   result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#   avg_a = np.average(result[:, :, 1])
#   avg_b = np.average(result[:, :, 2])
#   result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
#   result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
#   result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
#   return result
#
def show(final):
  print('display')
  cv2.imshow('Temple', final)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
#
# img = cv2.imread('A/0_3.jpeg')
# img_ = white_balance(img)
# show(img_)

# """function for automatic white balance using histogram matching"""

from PIL import Image
import colorcorrect.algorithm as cca
import numpy as np
import sys


def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.array(pimg)[:]
    # nimg.flags.writeable = True
    return nimg


def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

def crop_image_to_square(img):
    width, height = img.size
    if width > height:
        img = img.crop(((width - height) // 2, 0, (width + height) // 2, height))
    elif height > width:
        img = img.crop((0, (height - width) // 2, width, (height + width) // 2))
    return img

img = cv2.imread('A/0_3.jpeg')
# img = cv2.imread()
img_ = crop_image_to_square(img)

plt.imsave(img_)
show(img_)



# if __name__ == "__main__":
#     img = Image.open('A/0_1.jpeg')
#     # img.show()
#     AA = to_pil(cca.stretch(from_pil(img)))
#     # AA = to_pil(cca.grey_world(from_pil(img)))
#     # AA = to_pil(cca.retinex(from_pil(img)))
#     # AA = to_pil(cca.max_white(from_pil(img)))
#     # AA = to_pil(cca.retinex_with_adjust(cca.retinex(from_pil(img))))
#     # AA = to_pil(cca.standard_deviation_weighted_grey_world(from_pil(img), 20, 20))
#     # AA = to_pil(cca.standard_deviation_and_luminance_weighted_gray_world(from_pil(img), 20, 20))
#     # AA = to_pil(cca.luminance_weighted_gray_world(from_pil(img), 20, 20))
#     # AA = to_pil(cca.automatic_color_equalization(from_pil(img)))
#     AA.show()