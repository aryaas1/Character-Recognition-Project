import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import train
from train import labels

test_feat = []
norm_feat = []
components = []
img = None
D_index = []


def get_pic(str):
    global img
    img = io.imread(str)


def runtest(norm_feat, answer_index):
    th = 200
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
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
        if (maxc - minc >= 13) and (maxr - minr >= 13):
            test_feat.append(hu)

    for x in range(len(test_feat)):
        for y in range(7):
            test_feat[x][y] = (test_feat[x][y] - train.mean[y]) / train.std[y]


    D = cdist(test_feat, norm_feat)
    global D_index
    D_index = np.argsort(D, axis=1)


    def print_guess(index):
        char = chr(labels[D_index[index][0]] + 96)
        print(char, end=' ')


    print_guess(int(answer_index, 10))

