import cv2
import numpy as np

# Load citra
img = cv2.imread('daun.jpeg', cv2.IMREAD_GRAYSCALE)

# Konversi citra ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# Tentukan nilai threshold
thresh = 127

# Lakukan thresholding
_, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# Segmentasi citra
segmented_img = thresh_img.copy()
segmented_img[thresh_img == 0] = 0
segmented_img[thresh_img == 255] = gray[thresh_img == 255]

# Tampilkan hasil
cv2.imshow('Original', img)
cv2.imshow('Segmented', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()