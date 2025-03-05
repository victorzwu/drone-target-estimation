import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

# ==== CONFIGURATION ====
image_folder = "/Users/victorwu/Desktop/Github/drone-target-estimation/road_defect"
mask_output_folder = "/Users/victorwu/Desktop/Github/drone-target-estimation/road_defect_masks"
coco_json_path = os.path.join(image_folder, "_annotations.coco.json")
metadata_path = "/Users/victorwu/Desktop/Github/drone-target-estimation/metadata_sample_small.csv"

os.makedirs(mask_output_folder, exist_ok=True)

# ==== STEP 1: Convert COCO Annotations to PNG Masks ====
def convert_coco_to_masks(coco_json_path, image_folder, mask_output_folder):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_name = image_info["file_name"]
        width, height = image_info["width"], image_info["height"]

        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Find all annotations for this image
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:
                segmentation = annotation["segmentation"]
                for segment in segmentation:
                    polygon = np.array(segment).reshape((-1, 2))
                    polygon = polygon.astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 255)  # Fill defect area in white

        # Save the mask
        mask_filename = os.path.join(mask_output_folder, image_name.replace(".JPG", "_mask.png"))
        cv2.imwrite(mask_filename, mask)

    print("✅ Converted COCO segmentation to PNG masks.")

# Run COCO to Mask Conversion
convert_coco_to_masks(coco_json_path, image_folder, mask_output_folder)

# ==== STEP 2: Load Metadata ====
df = pd.read_csv(metadata_path)

# Extract drone positions
drone_positions = []
for i in range(len(df)):
    point_str = df["image_coords"][i].replace("SRID=4326;POINT(", "").replace(")", "")
    lon, lat = map(float, point_str.split())  # Convert to float
    alt = df["altitude"][i]
    drone_positions.append((lat, lon, alt))

print("✅ Loaded drone metadata.")

# ==== STEP 3: Load Images & Masks ====
image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".JPG")])
mask_filenames = sorted([f.replace(".JPG", "_mask.png") for f in image_filenames])

images = [cv2.imread(os.path.join(image_folder, filename)) for filename in image_filenames]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
masks = [cv2.imread(os.path.join(mask_output_folder, filename), cv2.IMREAD_GRAYSCALE) for filename in mask_filenames]

# Ensure masks are binary
masks = [cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1] for mask in masks]

print("✅ Loaded images and masks.")

# ==== STEP 4: Feature Matching with SIFT ====
sift = cv2.SIFT_create()
keypoints_descriptors = [sift.detectAndCompute(img, None) for img in gray_images]
keypoints = [kp_desc[0] for kp_desc in keypoints_descriptors]
descriptors = [kp_desc[1] for kp_desc in keypoints_descriptors]

# Filter keypoints inside the defect mask
def filter_keypoints_with_mask(keypoints, mask):
    return [kp for kp in keypoints if mask[int(kp.pt[1]), int(kp.pt[0])] > 0]

filtered_keypoints = [filter_keypoints_with_mask(kp, mask) for kp, mask in zip(keypoints, masks)]
filtered_descriptors = [sift.compute(gray_images[i], filtered_keypoints[i])[1] for i in range(len(images))]

print("✅ Extracted keypoints from defect region.")

# ==== STEP 5: Match Features Across Images ====
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = []
for i in range(len(images) - 1):
    matches.append(flann.knnMatch(filtered_descriptors[i], filtered_descriptors[i + 1], k=2))

good_matches = []
for match_set in matches:
    good_matches.append([m for m, n in match_set if m.distance < 0.75 * n.distance])

print(f"✅ Found {len(good_matches[0])} defect matches.")

# def visualize_feature_matching(img1, kp1, img2, kp2, matches, title="Feature Matching"):
#     """
#     Draws and displays SIFT feature matches between two images.
#     """
#     match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
#     plt.figure(figsize=(12, 6))
#     plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

# # Show feature matching between the first two images
# visualize_feature_matching(
#     images[0], filtered_keypoints[0], 
#     images[1], filtered_keypoints[1], 
#     good_matches[0], 
#     title="SIFT Feature Matching (Defect Region)"
# )

# ==== STEP 6: Estimate Real-World Position ====
def get_defect_keypoints(matches, keypoints1, keypoints2):
    defect_points1 = []
    defect_points2 = []
    for match in matches:
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt
        defect_points1.append(kp1)
        defect_points2.append(kp2)
    return np.array(defect_points1), np.array(defect_points2)

defect_points1, defect_points2 = get_defect_keypoints(good_matches[0], filtered_keypoints[0], filtered_keypoints[1])

def compute_displacement(defect_points1, defect_points2):
    shifts = np.linalg.norm(defect_points1 - defect_points2, axis=1)
    return np.mean(shifts)

baseline_distance = 5  # Approximate drone movement in meters
focal_length = 800  # Camera focal length in pixels

parallax_shift = compute_displacement(defect_points1, defect_points2)
defect_depth = (focal_length * baseline_distance) / parallax_shift if parallax_shift else float('inf')

defect_lat = (drone_positions[0][0] + drone_positions[1][0]) / 2
defect_lon = (drone_positions[0][1] + drone_positions[1][1]) / 2

print(f"✅ Estimated Defect Location: Latitude {defect_lat}, Longitude {defect_lon}, Depth {defect_depth:.2f}m")

# Earth's radius in meters
EARTH_RADIUS = 6371000

def refine_defect_location(drone_lat, drone_lon, defect_depth, gimbal_pitch, heading):
    """
    Refines the defect's GPS location using depth, camera angles, and drone heading.
    """

    # Convert angles to radians
    gimbal_pitch_rad = np.radians(gimbal_pitch)
    heading_rad = np.radians(heading)

    # Calculate ground distance using gimbal pitch angle
    ground_distance = defect_depth * np.tan(gimbal_pitch_rad)

    # Convert ground distance to (X, Y) offsets using drone heading
    delta_x = ground_distance * np.sin(heading_rad)  # East-West offset (meters)
    delta_y = ground_distance * np.cos(heading_rad)  # North-South offset (meters)

    # Convert (X, Y) offset to latitude/longitude
    delta_lat = (delta_y / EARTH_RADIUS) * (180 / np.pi)
    delta_lon = (delta_x / (EARTH_RADIUS * np.cos(np.radians(drone_lat)))) * (180 / np.pi)

    # Compute refined latitude and longitude
    refined_lat = drone_lat + delta_lat
    refined_lon = drone_lon + delta_lon

    return refined_lat, refined_lon

# === Step 1: Get Drone Metadata for the First Image ===
drone_lat, drone_lon, _ = drone_positions[0]  # Use first drone position
gimbal_pitch = df["gimbal_pitch"][0]  # Extract gimbal pitch angle from metadata
heading = df["heading"][0]  # Extract drone heading from metadata

# === Step 2: Compute Refined Defect Location ===
refined_lat, refined_lon = refine_defect_location(drone_lat, drone_lon, defect_depth, gimbal_pitch, heading)

print(f"✅ Refined Defect Location: Latitude {refined_lat}, Longitude {refined_lon}, Depth {defect_depth:.2f}m")


# ==== STEP 7: Visualize on a Map ====
defect_map = folium.Map(location=[refined_lat, refined_lon], zoom_start=18)

folium.Marker(
    location=[refined_lat, refined_lon],
    popup=f"Estimated Defect\nLat: {defect_lat}, Lon: {defect_lon}\nDepth: {defect_depth:.2f}m",
    icon=folium.Icon(color="red", icon="info-sign"),
).add_to(defect_map)

for lat, lon, _ in drone_positions:
    folium.Marker(
        location=[lat, lon],
        popup=f"Drone Position\nLat: {lat}, Lon: {lon}",
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(defect_map)

defect_map.save("/Users/victorwu/Desktop/Github/drone-target-estimation/defect_map.html")
print("✅ Saved defect map: defect_map.html")