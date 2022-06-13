#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

from distutils.log import debug
import numpy as np
import cv2 as cv
from pathlib import Path
import sys
# built-in modules
import os
import glob
import argparse

class FileSource():
    def __init__(self, input_dir, ext='bmp'):
        search = f'{input_dir}/*'
        self.files = sorted(glob.glob(search))
        self.index = 0

    def read(self, backwards=False):
        if backwards:
            self.index -= 1
        else:
            self.index += 1

        if self.index >= len(self.files):
            self.index = 0
        elif self.index < 0:
            self.index = len(self.files) - 1

        print(f'Index: {self.index} File: {self.files[self.index]}')
        img = cv.imread(self.files[self.index])
        return True, img

    def isOpened(self):
        if len(self.files) > 0:
            return True
        return False

class BlobDetector():
    def __init__(self):
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        ########################################Blob Detector##############################################

        # Setup SimpleBlobDetector parameters.
        blobParams = cv.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 8
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 2000   # minArea may be adjusted to suit for your experiment
        blobParams.maxArea = 4500   # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.1

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.87

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.01

        # Create a detector with the parameters
        self.blobDetector = cv.SimpleBlobDetector_create(blobParams)

        ###################################################################################################

        ###################################################################################################

        # Original blob coordinates, supposing all blobs are of z-coordinates 0
        # And, the distance between every two neighbour blob circle centers is 72 centimetres
        # In fact, any number can be used to replace 72.
        # Namely, the real size of the circle is pointless while calculating camera calibration parameters.
        self.objp = np.zeros((55, 3), np.float32)
        index = 0
        for x in range(0, 11):
            for y in range(0, 5):
                self.objp[index] = (x * 33, y * 33, 0)
                index += 1
        ###################################################################################################

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        self.found = 0

    def _verify_gray(self, img):
        if len(img.shape) > 2:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray

    def detect(self, img):
        gray = self._verify_gray(img)
        gray = cv.rotate(gray, cv.ROTATE_180)
        keypoints = self.blobDetector.detect(gray) # Detect blobs.
        im_with_keypoints = cv.drawKeypoints(gray, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findCirclesGrid(gray, (5,11), blobDetector = self.blobDetector, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid
        print(f'Number of keypoints: {len(keypoints)}')

        if ret == True:
            print('Found some corners (small miracle)')
            self.objpoints.append(self.objp)  # Certainly, every loop objp is the same, in 3D.

            corners2 = cv.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), self.criteria)    # Refines the corner locations.
            self.imgpoints.append(corners2)

            # Draw and display the corners.
            im_with_keypoints = cv.drawChessboardCorners(gray, (5,11), corners2, ret)
            self.found += 1
        else:
            print('No valid corners found')

        return im_with_keypoints

    def calibrate(self, img):
        gray = self._verify_gray(img)
        h, w = gray.shape
        print(f'Num objpoints: {len(self.objpoints)}  Num imgpoints: {len(self.imgpoints)}')
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, (w, h), None, None)
        print(f'Ret: {ret} mtx: {mtx} dist: {dist}')


def main():
    parser = argparse.ArgumentParser(description='OpenCV Acircle camera cal')
    parser.add_argument('-i', '--input', type=str, default='',
                        help='Specify input directory')
    parser.add_argument('-o', '--output', type=str, default='calout',
                        help='Output debug files')
    args = parser.parse_args()

    debug_dir = Path(args.output)
    if not debug_dir.exists():
        debug_dir.mkdir()

    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    #pattern_points *= square_size

    bd = BlobDetector()


    def show(img, winname='img'):
        small = cv.resize(img, (1280,720))
        cv.imshow(winname, small)
        return cv.waitKey(1)

    fs = FileSource(args.input)
    ret, img = fs.read()

    update = False
    while True:
        key = show(img)
        if key == ord('2'):
            ret, img = fs.read()
            update = True
        elif key == ord('1'):
            ret, img = fs.read(True)
            update = True
        elif key == ord('q'):
            break
        elif key == ord('c'):
            bd.calibrate(img)
        elif key == ord('u'):
            update = True

        if update:
            update = False
            drawimg = bd.detect(img)
            show(drawimg, 'blobs')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
