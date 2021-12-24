#!/usr/bin/env python
import json
import cv2
import argparse
import os
from os import path
import numpy as np

# Parse args
parser = argparse.ArgumentParser(description='Distortion removal')

parser.add_argument(
    'calib_json',
    metavar='JSON',
    type=str,
    help='JSON file containing calibration parameters'
)

parser.add_argument(
    'images',
    metavar='IMAGE',
    nargs='+',
    type=str,
    help='Image to undistort'
)

parser.add_argument(
    '--output',
    metavar='DIRECTORY',
    default='',
    type=str,
    help='Output path'
)

parser.add_argument(
    '--crop',
    action='store_true',
    help='Output cropped images'
)

args = parser.parse_args()


def main():
    # Read calibration data
    with open(args.calib_json, 'rt') as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coefs = np.array(calib['distortion_coefficients'])

    for img in args.images:
        # Read image
        img_name = os.path.basename(img)
        img = cv2.imread(img)
        h, w = img.shape[:2]

        # Get undistortion matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coefs,
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            img,
            camera_matrix,
            dist_coefs,
            None,
            newcameramtx
        )

        # Optionally crop
        if args.crop:
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]

        # Save image
        cv2.imwrite(
            path.join(args.output, 'undistort_' + img_name),
            undistorted
        )


if __name__ == '__main__':
    main()
