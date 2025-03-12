import os
import cv2
import json
import numpy as np
import pyproj
import exifread

# ---------- CONFIGURATION ----------
# JSON containing points (cx, cy) per image
point_json_path = "Six-Locations/pos-6/position.json"
filtered_image_folder = "Six-Locations/pos-6/images"  # Folder with images
output_folder = "Six-Locations/pos-6/output"  # Folder for visualization
json_output_path = "Six-Locations/pos-6/output/coordinates.json"  # JSON file output
os.makedirs(output_folder, exist_ok=True)

# Load JSON file containing marked points
with open(point_json_path, 'r') as f:
    point_data = json.load(f)

# Define coordinate transformation (WGS84 to ECEF)
wgs84 = pyproj.CRS("EPSG:4326")  # Latitude, Longitude
ecef = pyproj.CRS("EPSG:4978")  # Earth-Centered, Earth-Fixed
transformer_to_ecef = pyproj.Transformer.from_crs(wgs84, ecef, always_xy=True)
transformer_to_wgs84 = pyproj.Transformer.from_crs(ecef, wgs84, always_xy=True)


# ---------- FUNCTION: Extract Drone Metadata ----------
def get_exif_metadata(image_path):
    """Extracts GPS coordinates, altitude, focal length, yaw, pitch, roll from EXIF metadata."""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    def dms_to_degrees(dms):
        """Convert EXIF GPS DMS to decimal degrees."""
        return float(dms[0].num) / float(dms[0].den) + \
            float(dms[1].num) / (60 * float(dms[1].den)) + \
            float(dms[2].num) / (3600 * float(dms[2].den))

    # Extract GPS
    lat = dms_to_degrees(
        tags['GPS GPSLatitude'].values) if 'GPS GPSLatitude' in tags else None
    lon = dms_to_degrees(
        tags['GPS GPSLongitude'].values) if 'GPS GPSLongitude' in tags else None
    if 'GPS GPSLatitudeRef' in tags and tags['GPS GPSLatitudeRef'].values == 'S':
        lat = -lat
    if 'GPS GPSLongitudeRef' in tags and tags['GPS GPSLongitudeRef'].values == 'W':
        lon = -lon
    alt = float(tags['GPS GPSAltitude'].values[0].num) / float(
        tags['GPS GPSAltitude'].values[0].den) if 'GPS GPSAltitude' in tags else None

    # Extract Focal Length
    focal_length = float(tags['EXIF FocalLength'].values[0].num) / \
        float(
            tags['EXIF FocalLength'].values[0].den) if 'EXIF FocalLength' in tags else 24.0

    # Extract Yaw, Pitch, Roll (defaults to 0 if missing)
    yaw = float(tags.get("FlightYawDegree", [0])[0])
    pitch = float(tags.get("FlightPitchDegree", [0])[0])
    roll = float(tags.get("FlightRollDegree", [0])[0])

    return {
        "lat": lat, "lon": lon, "alt": alt or 0,
        "focal_length": focal_length, "yaw": yaw, "pitch": pitch, "roll": roll
    }


# ---------- FUNCTION: Convert Pixel → World Coordinates ----------
def pixel_to_world(image_metadata, pixel_coords, image_shape):
    """Convert pixel coordinates to real-world coordinates using the Pinhole Camera Model."""
    if image_metadata["focal_length"] is None:
        return []  # Skip images with missing focal length

    fx = fy = image_metadata["focal_length"] * \
        image_shape[1]  # Focal length in pixels
    x, y = image_shape[1] / 2, image_shape[0] / 2  # Camera center

    real_world_coords = []
    for px, py in pixel_coords:
        x_cam = (px - x) / fx
        y_cam = (py - y) / fy
        z_cam = 1  # Assume ground-plane intersection

        # Convert to GPS
        lat_offset = y_cam * 0.00001
        lon_offset = x_cam * 0.00001
        real_world_coords.append(
            (image_metadata["lat"] + lat_offset, image_metadata["lon"] + lon_offset))

    return real_world_coords


# ---------- PROCESS EACH IMAGE ----------
tree_positions_by_image = {}

for image_name, coordinates in point_data.items():
    image_path = os.path.join(filtered_image_folder, image_name)

    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Image {image_name} not found!")
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Skipping {image_name}, unable to load image.")
        continue

    metadata = get_exif_metadata(image_path)

    # Skip if GPS coordinates are missing
    if metadata["lat"] is None or metadata["lon"] is None:
        continue

    pixel_points = [(coordinates["x"], coordinates["y"])]

    # Convert to world coordinates using yaw, pitch, roll
    world_positions = pixel_to_world(metadata, pixel_points, image.shape)

    # Store positions
    tree_positions_by_image[image_name] = world_positions

    # Mark image with detected point
    for (px, py) in pixel_points:
        cv2.circle(image, (int(px), int(py)), radius=10,
                   color=(0, 255, 0), thickness=-1)

    cv2.imwrite(os.path.join(output_folder, image_name), image)

# Save tree positions
with open(json_output_path, "w") as f:
    json.dump(tree_positions_by_image, f, indent=4)

print(f"✅ Tree positions saved to {json_output_path}")
