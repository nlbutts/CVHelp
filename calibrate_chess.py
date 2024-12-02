import cv2
import numpy as np
import glob

# Define the checkerboard dimensions (inner corners)
#CHECKERBOARD = (9, 6)  # Change to match your checkerboard's inner corners
CHECKERBOARD = (10, 7)  # Change to match your checkerboard's inner corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Lists to store object points and image points
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane

# Load all checkerboard images
images = glob.glob('/mnt/hgfs/ztemp/high_speed_cal/*.bmp')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)  # Append 3D points
        # Refine the corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)
    else:
        print(f'{fname} found 0 corners')
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration
print(f'Calibrating with {len(imgpoints)} image points')
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print calibration results
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

# Save the calibration results
np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

# Optional: Undistort an example image
example_img = cv2.imread(images[0])  # Update path
h, w = example_img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(example_img, camera_matrix, dist_coeffs, None, new_camera_mtx)

# Crop the image (optional)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

cv2.imshow('Original Image', example_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
