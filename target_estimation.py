import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the images
image_folder = "road_defect"
image_filenames = ["DJI_0128.JPG", "DJI_0129.JPG", "DJI_0130.JPG", "DJI_0131.JPG"]

# Load images
images = [cv2.imread(os.path.join(image_folder, filename)) for filename in image_filenames]

# Convert images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints_descriptors = [sift.detectAndCompute(img, None) for img in gray_images]

# Extract keypoints and descriptors
keypoints = [kp_desc[0] for kp_desc in keypoints_descriptors]
descriptors = [kp_desc[1] for kp_desc in keypoints_descriptors]

# FLANN-based matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match features between consecutive images
matches = []
for i in range(len(images) - 1):
    matches.append(flann.knnMatch(descriptors[i], descriptors[i + 1], k=2))

# Apply Lowe’s ratio test to filter good matches
good_matches = []
for match_set in matches:
    good_matches.append([m for m, n in match_set if m.distance < 0.75 * n.distance])

# Draw matches for visualization
def draw_matches(img1, kp1, img2, kp2, matches):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img)
    plt.show()

# Visualize feature matching for the first image pair (for debugging)
draw_matches(images[0], keypoints[0], images[1], keypoints[1], good_matches[0])

print(f"Feature matching completed for {len(images)} images.")