import argparse
import os
import cv2
import numpy as np

"""This script calculates the MSE between two images.
"""

def print_stats(img):
    temp = img.flatten()
    meanv = np.mean(temp)
    stddev = np.std(temp)
    maxv = np.max(temp)
    minv = np.min(temp)
    print(f'{meanv} {stddev} {maxv} {minv}')

parser = argparse.ArgumentParser(
                    prog='Histogram Generator',
                    description='Plots histograms all all images')

parser.add_argument('inputs', nargs=2, help='Two files to compare')

args = parser.parse_args()

files = args.inputs
print(files)
exts = []
for file in files:
    info = file.split('.')
    exts.append(info[-1].lower())

if exts[0] != exts[1]:
    print('File type are different. They must be the same')
    exit(-1)

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.png')

if files[0].lower().endswith(image_extensions):
    print('The file type in an image')

    img1 = cv2.imread(files[0])
    img2 = cv2.imread(files[1])
    h, w, ch = img1.shape

    img1 = img1.astype('int')
    img2 = img2.astype('int')

    diff = img1 - img2
    diff = diff * diff
    mse = np.sum(diff)
    print(f'{w}x{h}x{ch}')
    mse /= h * w * ch
    print(f'{files[0]} {files[1]} MSE: {mse}')

elif exts[0] == 'bin' or exts[0] == 'raw':
    print('The file type is a raw binary')
    img1 = np.fromfile(files[0], 'uint8').astype('int')
    img2 = np.fromfile(files[1], 'uint8').astype('int')
    diff = img1 - img2
    diff = diff * diff
    mse = np.sum(diff)
    mse /= (1984 * 1204 * 1)
    print(f'Using uint8 datatype {files[0]} {files[1]} MSE: {mse}')
    print_stats(img2)

    img1 = np.fromfile(files[0], 'uint16').astype('int') >> 4
    img2 = np.fromfile(files[1], 'uint16').astype('int') >> 4
    diff = img1 - img2
    diff = diff * diff
    mse = np.sum(diff)
    mse /= (1984 * 1204 * 2)
    print(f'Using uint16 datatype {files[0]} {files[1]} MSE: {mse}')
    print_stats(img2)
