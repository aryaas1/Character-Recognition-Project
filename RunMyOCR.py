import pickle
import train
from train import features, labels
import test
from test import D_index, components


test_image = input("Enter name of the image file : ")
answer_index = input("Enter index of character you want to guess (0-69) : ")


# def parse_gt_file(file_path):
#     gt = {}
#     with open(file_path, 'rb') as file:
#         gt = pickle.load(file)
#
#     classes = gt.get('classes', gt[b'classes'])
#     locations = gt.get('locations', gt[b'locations'])
#     return classes, locations


#classes, locations = parse_gt_file('test_gt_py3.pkl')


def execute_train(str):
    train.get_images(str, False)
    train.get_features()

TRAINING_IMAGES = [
    'images/a.bmp',
    'images/d.bmp',
    'images/m.bmp',
    'images/n.bmp',
    'images/o.bmp',
    'images/p.bmp',
    'images/q.bmp',
    'images/r.bmp',
    'images/u.bmp',
]

for i in TRAINING_IMAGES:
    execute_train(i)

norm_features = train.norm(features)


def execute_test(str, ans_ind):
    test.get_pic(str)
    test.runtest(norm_features, ans_ind)


execute_test(test_image, answer_index)

