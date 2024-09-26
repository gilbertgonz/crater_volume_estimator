import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global list to store clicked points
clicked_points = []

# Mouse callback function to capture clicked points
def capture_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # Draw a small circle at the clicked point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle
        cv2.imshow('Image', image)

# Load the RGB image
image = cv2.imread('data/output.jpg')
original_img = image.copy()
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', capture_point)

# Wait for the user to click points
print("Click on the image to define the circumference of the pothole. Press 'q' to finish.")
while True:
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to finish clicking
        break

cv2.destroyAllWindows()

# Convert clicked points to a NumPy array for mask creation
clicked_points_np = np.array(clicked_points, dtype=np.int32)

# Create a binary mask for the pothole circumference
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [clicked_points_np], 255)

# Define depths for different areas
depth_value_deep = 1.2  # Depth at the bottom of the pothole (in meters)
depth_value_shallow = 1.0  # Depth outside the pothole

# Create a depth map initialized to zero
depth_map = np.full_like(mask, depth_value_shallow, dtype=np.float32)

# Fill the depth map with deeper values inside the circumference
depth_map[mask > 0] = depth_value_deep  # Set depth inside the mask

# Calculate the centroid of the clicked points to determine the center of the pothole
if len(clicked_points) > 0:
    centroid_x = np.mean(clicked_points_np[:, 0])
    centroid_y = np.mean(clicked_points_np[:, 1])
    
    # Calculate the radius based on the maximum distance from the centroid to the clicked points
    radius = np.max(np.sqrt((clicked_points_np[:, 0] - centroid_x) ** 2 + (clicked_points_np[:, 1] - centroid_y) ** 2))

    # # Create a gradient depth effect
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i, j] > 0:  # Only adjust depth where the mask is applied
    #             distance_to_center = cv2.pointPolygonTest(clicked_points_np, (j, i), True)
    #             # Decrease depth as we move towards the center of the pothole
    #             if radius > 0:  # Ensure we don't divide by zero
    #                 depth_map[i, j] = depth_value_deep + (radius - distance_to_center) * (depth_value_deep / radius)

    # Add noise to depth map
    noise = np.random.normal(loc=0.0, scale=0.005, size=depth_map.shape)
    depth_map += noise

# Combine RGB and Depth Data
output_data = {}

# Fill the RGBA channels
output_data['rgb'] = original_img
output_data['depth'] = depth_map

# Save the output as a .npy file
np.save('noisy_synthetic_rgbd_output.npy', output_data)

# Now access the RGB and depth data
rgb_image = output_data['rgb']  # Access the RGB image
depth_image = output_data['depth']  # Access the depth map

# Normalize depth image for better visualization
depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
depth_image_normalized = (depth_image_normalized * 255).astype(np.uint8)  # Scale to 0-255

# Create a figure to display the images
plt.figure(figsize=(10, 5))

# Plot RGB image
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('RGB Image')
plt.axis('off')

# Plot Depth image
plt.subplot(1, 2, 2)
plt.imshow(depth_image_normalized, cmap='gray')
plt.title('Depth Image')
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()
