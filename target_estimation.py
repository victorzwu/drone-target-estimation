import cv2
import numpy as np
import json
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from haversine import haversine

# Load two images
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Use FLANN-based matcher to find feature matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
matched_objects = []
for match in good_matches:
    pt1 = kp1[match.queryIdx].pt  # (x, y) in image1
    pt2 = kp2[match.trainIdx].pt  # (x, y) in image2
    matched_objects.append((pt1, pt2))

# Example GPS coordinates of drone locations
drone1_gps = (34.040314, -118.535265)
drone2_gps = (34.040400, -118.535360)

# Compute baseline distance between drones
baseline_distance = haversine(drone1_gps, drone2_gps)  # in km

# Triangulation function
def estimate_depth_from_shift(pt1, pt2, baseline):
    shift = abs(pt1[0] - pt2[0])  # X-coordinate difference in pixels
    focal_length = 800  # Hypothetical focal length in pixels
    depth = (baseline * focal_length) / shift  # Depth estimation (scaled)
    return depth

# Convert depth & image position to GPS
def convert_to_gps(drone_gps, image_point, depth):
    pixel_offset_meters = depth * 0.1  # Convert arbitrary depth units to meters
    lat_offset = pixel_offset_meters / 111320  # Convert meters to latitude offset
    lon_offset = pixel_offset_meters / (111320 * np.cos(np.radians(drone_gps[0])))
    return (drone_gps[0] + lat_offset, drone_gps[1] + lon_offset)

# Process each matched object
final_detections = []
for obj1, obj2 in matched_objects:
    depth = estimate_depth_from_shift(obj1, obj2, baseline_distance)
    object_gps = convert_to_gps(drone1_gps, obj1, depth)
    final_detections.append(object_gps)

# Apply DBSCAN clustering to remove duplicate detections
coords = np.array(final_detections)
clustering = DBSCAN(eps=0.0001, min_samples=2, metric='haversine').fit(np.radians(coords))
unique_detections = coords[np.unique(clustering.labels_, return_index=True)[1]]

# Save results as GeoJSON
def save_geojson(detections, filename="detections.geojson"):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {}
            }
            for lat, lon in detections
        ]
    }
    with open(filename, "w") as f:
        json.dump(geojson, f, indent=4)

# Save the final geolocated objects
save_geojson(unique_detections)
