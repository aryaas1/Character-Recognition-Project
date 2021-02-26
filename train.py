import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

features = []
labels = []
mean = []
std = []
img = None
File = None


def get_images(str, show):
    global img
    img = io.imread(str)
    global file
    file = str
    if show:
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()


def binary(display):
    th = 200
    img_binary = (img < th).astype(np.double)
    if display:
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    return img_binary


def lab(display):
    img_binary = binary(False)
    img_label = label(img_binary, background=0)
    if display:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
    return img_label
    print(np.amax(img_label))


def get_features():
    imglabel = lab(False)
    regions = regionprops(imglabel)
    img_binary = binary(False)
    io.imshow(img_binary)
    ax = plt.gca()
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        h = maxc - minc
        w = maxr - minr
        if (h > 10) and (w > 10):
            global features
            features.append(hu)
            labels.append(char_label(file))
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
                                       fill=False, edgecolor='red', linewidth=1))

    # ax.set_title('Bounding Boxes')
    # io.show()


def norm(features):
    for x in range(7):
        res = [sub[x] for sub in features]
        mt = np.mean(res)
        mean.append(mt)

    for x in range(7):
        res = [sub[x] for sub in features]
        st = np.std(res)
        std.append(st)

    for x in range(len(features)):
        for y in range(7):
            features[x][y] = (features[x][y] - mean[y]) / std[y]

    return features


def char_label(str):
    return ord(str[7]) - 96



