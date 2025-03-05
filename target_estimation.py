import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---- Step 1: Load Metadata ----
metadata_path = "metadata_sample_small.csv"
df = pd.read_csv(metadata_path)

# Extract drone positions (latitude, longitude, altitude)
drone_positions = []
for i in range(len(df)):
    # Extract coordinates from the POINT format
    point_str = df["image_coords"][i].replace("SRID=4326;POINT(", "").replace(")", "")
    lon, lat = map(float, point_str.split())  # Convert to float
    alt = df["altitude"][i]
    drone_positions.append((lat, lon, alt))

print("Loaded drone metadata successfully.")

# ---- Step 2: Load Images ----
image_folder = "road_defect"
image_filenames = ["DJI_0128.JPG", "DJI_0129.JPG", "DJI_0130.JPG", "DJI_0131.JPG"]
images = [cv2.imread(os.path.join(image_folder, filename)) for filename in image_filenames]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# ---- Step 3: Detect Features Using SIFT ----
sift = cv2.SIFT_create()
keypoints_descriptors = [sift.detectAndCompute(img, None) for img in gray_images]
keypoints = [kp_desc[0] for kp_desc in keypoints_descriptors]
descriptors = [kp_desc[1] for kp_desc in keypoints_descriptors]

# ---- Step 4: Match Features Across Images ----
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = []
for i in range(len(images) - 1):
    matches.append(flann.knnMatch(descriptors[i], descriptors[i + 1], k=2))

# Apply Lowe’s ratio test to filter good matches
good_matches = []
for match_set in matches:
    good_matches.append([m for m, n in match_set if m.distance < 0.75 * n.distance])

# ---- Step 5: Extract Keypoints for Defect ----
def get_defect_keypoints(matches, keypoints1, keypoints2):
    """ Extracts matched keypoints assumed to belong to the defect. """
    defect_points1 = []
    defect_points2 = []
    
    for match in matches:
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt
        defect_points1.append(kp1)
        defect_points2.append(kp2)

    return np.array(defect_points1), np.array(defect_points2)

# ---- Step 6: Compute Parallax Shift ----
def compute_displacement(defect_points1, defect_points2):
    """ Computes the displacement (parallax shift) of the defect across images. """
    shifts = np.linalg.norm(defect_points1 - defect_points2, axis=1)
    return np.mean(shifts)  # Average shift across all detected points

# ---- Step 7: Estimate Depth Using Triangulation ----
def estimate_depth(displacement, baseline_distance, focal_length):
    """ Uses a simple triangulation formula to estimate depth. """
    if displacement == 0:
        return float('inf')  # Object is at infinity
    return (focal_length * baseline_distance) / displacement

# ---- Step 8: Convert to Real-World Coordinates ----
def estimate_real_world_position(drone_pos1, drone_pos2, defect_depth):
    """ Estimate defect location based on triangulated depth. """
    lat1, lon1, _ = drone_pos1
    lat2, lon2, _ = drone_pos2

    # Approximate the midpoint as a starting estimate
    defect_lat = (lat1 + lat2) / 2
    defect_lon = (lon1 + lon2) / 2

    return defect_lat, defect_lon

# ---- Step 9: Compute Defect's Real-World Position ----
defect_points1, defect_points2 = get_defect_keypoints(good_matches[0], keypoints[0], keypoints[1])
parallax_shift = compute_displacement(defect_points1, defect_points2)

# Define parameters
baseline_distance = 5  # Approximate drone movement between captures (meters)
focal_length = 800  # Approximate focal length (in pixels)

# Estimate defect depth
defect_depth = estimate_depth(parallax_shift, baseline_distance, focal_length)

# Estimate real-world position
defect_lat, defect_lon = estimate_real_world_position(drone_positions[0], drone_positions[1], defect_depth)

# ---- Step 10: Output Results ----
print(f"Estimated Defect Location:")
print(f"Latitude: {defect_lat}, Longitude: {defect_lon}, Depth: {defect_depth:.2f} meters")

# ---- Step 11: Visualize Feature Matching (Optional) ----
def draw_matches(img1, kp1, img2, kp2, matches):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(match_img)
    plt.show()

# Show feature matching for debugging
draw_matches(images[0], keypoints[0], images[1], keypoints[1], good_matches[0])