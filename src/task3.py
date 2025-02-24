# portions of this code were copy and pasted from
# task2.py which is sourced from Adam Czajka, Jin Huang, September 2019

import os
import cv2
import numpy as np
from skimage import measure
from skimage.measure import label, regionprops
from sys import platform as sys_pf
import warnings
warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

# Read the image into grayscale
currentDirectory = os.path.dirname(__file__)
parentDirectory = os.path.join(currentDirectory, "..")
savePath = os.path.join(parentDirectory, "output", "task3")
imagePath = os.path.join(parentDirectory, "data", "pills.png")
print(imagePath)
sampleColor = cv2.imread(imagePath)
sample = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

sample_small = cv2.resize(sample, (640, 480))
grayScalePath = os.path.join(savePath, "greyScale.png")
cv2.imwrite(grayScalePath, sample_small)
cv2.imshow('Grey scale image',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sample_small = cv2.resize(binary_image, (640, 480))
otsuScalePath = os.path.join(savePath, "otsu.png")
cv2.imwrite(otsuScalePath, sample_small)
cv2.imshow('Image after Otsu''s thresholding',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()


# morphological operations
kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# morphological opening:
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernelOpen)
# morphological closing:
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernelClose)

sample_small = cv2.resize(binary_image, (640, 480))
morphologicalScalePath = os.path.join(savePath, "morphological.png")
cv2.imwrite(morphologicalScalePath, sample_small)
cv2.imshow('Image after morphological operations',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()


# finding the connected pixels and calculate features
labels = measure.label(binary_image, connectivity = 2, background = 0)
properties = measure.regionprops(labels)
features = np.zeros((len(properties), 2))

for prop in properties:
    minr, minc, maxr, maxc = prop.bbox
    # If the region touches the border, skip it
    if (minr == 0 or minc == 0 or maxr == labels.shape[0] or maxc == labels.shape[1]):
        continue


for i in range(0, len(properties)):
    features[i, 0] = properties[i].perimeter

# show histogram of perimeters
perimeters = [prop.perimeter for prop in properties]

# Plot a histogram of the perimeters
plt.hist(perimeters, bins = 20) 
plt.xlabel("Perimeter")
plt.ylabel("Count")
plt.show()

threshold = 100
circlePills = 0
ovelPills = 0

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(sampleColor, cv2.COLOR_BGR2RGB))

for i in range(0, len(properties)):
    if (features[i, 0] > threshold):
        circlePills = circlePills + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.g', markersize=15)

    if (features[i, 0] < threshold):
        ovelPills = ovelPills + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.b', markersize=15)

plt.show()

print("I found %d circle pills, and %d ovel pills." % (circlePills, ovelPills))
