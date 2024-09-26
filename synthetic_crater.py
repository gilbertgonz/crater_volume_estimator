from sklearn.linear_model import RANSACRegressor
import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth, pix_per_m_x, pix_per_m_y, cx, cy):
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calculate the 3D coordinates
    z = depth
    x_3d = (x - cx) * z / pix_per_m_x
    y_3d = (y - cy) * z / pix_per_m_y
    
    return np.column_stack((x_3d.flatten(), y_3d.flatten(), z.flatten()))

def plot(point_cloud, crater_mask, a, b, c):
    pcd = o3d.geometry.PointCloud()

    # Assign colors based on crater points
    colors = np.zeros(point_cloud.shape)  # Default to black
    colors[crater_mask] = [1, 0, 0]    # Red for crater points
    colors[~crater_mask] = [0, 0, 1]   # Blue for non-crater points
    
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a grid for the ground plane
    x_range = np.linspace(min(point_cloud[:, 0]), max(point_cloud[:, 0]), 10)
    y_range = np.linspace(min(point_cloud[:, 1]), max(point_cloud[:, 1]), 10)
    X, Y = np.meshgrid(x_range, y_range)
    Z = a * X + b * Y + c  # Compute Z values for the plane
    plane_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create a mesh for the ground plane
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
    plane_pcd.paint_uniform_color([0, 0, 0])  # Light gray for the plane

    # Visualize
    o3d.visualization.draw_geometries([pcd, plane_pcd], window_name='3D Plot')

def insert_crater(depth_data, crater_depth, crater_dim_meters, pixels_per_meter_x, pixels_per_meter_y):
    # Crater dimensions in pixels
    crater_dim_pixels_x = int(crater_dim_meters[0] * pixels_per_meter_x)
    crater_dim_pixels_y = int(crater_dim_meters[1] * pixels_per_meter_y)

    # Calculate the starting and ending indices to center the crater
    start_row = (depth_data.shape[0] - crater_dim_pixels_y) // 2
    end_row = start_row + crater_dim_pixels_y
    start_col = (depth_data.shape[1] - crater_dim_pixels_x) // 2
    end_col = start_col + crater_dim_pixels_x

    # Update the depth data for the crater area
    depth_data[start_row:end_row, start_col:end_col] = crater_depth

def generate_pointcloud(params):
    # Assume camera intrinsics 
    cam_width = params["cam_dim"][0]
    cam_height = params["cam_dim"][1]
    pixels_per_meter_x = cam_width / params["ground_plane_details"]['dim_meters'][0]
    pixels_per_meter_y = cam_height / params["ground_plane_details"]['dim_meters'][1]
    cx = (cam_width - 1) / 2
    cy = (cam_height - 1) / 2

    # Initialize a 3D ground plane
    depth_data = np.full((cam_width, cam_height), params["ground_plane_details"]['depth_meters'], dtype=np.float32)

    # Insert crater into depth data
    insert_crater(depth_data, params["crater_details"]['depth_meters'], params["crater_details"]['dim_meters'], pixels_per_meter_x, pixels_per_meter_y)

    # Apply noise
    noise = np.random.normal(loc=0.0, scale=params["ground_plane_details"]['depth_meters'] * params["depth_noise"], size=depth_data.shape)
    depth_data += noise

    point_cloud = depth_to_pointcloud(depth_data, pixels_per_meter_x, pixels_per_meter_y, cx, cy)
    
    # Calculate pixel area based on depth
    pixel_areas = (point_cloud[:, 2] / pixels_per_meter_x) * (point_cloud[:, 2] / pixels_per_meter_y)

    # Convert the synthetic depth map to a point cloud
    return point_cloud, pixel_areas

def find_ground_plane(point_cloud):
    # Use RANSAC to fit a plane to the point cloud
    X = point_cloud[:, :2] 
    Y = point_cloud[:, 2]

    # Fit a plane using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X, Y)

    # Extract the ground plane (plane equation: Z = a*X + b*Y + c)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    return a, b, c

def main():
    params = {
        "cam_dim": (120, 160),
        "depth_noise": 0.01,  # https://www.intel.com/content/www/us/en/support/articles/000026260/emerging-technologies/intel-realsense-technology.html
        "ground_plane_details": {
            "depth_meters": 1.0,
            "dim_meters": (3, 4)
        },
        "crater_details": {
            "depth_meters": 1.2,
            "dim_meters": (1, 1)
        }
    }

    # Generate point cloud
    point_cloud, pixel_areas = generate_pointcloud(params)

    # Find ground plane using RANSAC
    a, b, c = find_ground_plane(point_cloud)

    # Compute the distance of each point to the ground plane
    distances = np.abs(a * point_cloud[:, 0] + b * point_cloud[:, 1] - point_cloud[:, 2] + c) / np.sqrt(a**2 + b**2 + 1)

    # Identify cavities
    crater_threshold = 0.1  # meters
    crater_mask = distances > crater_threshold

    # Calculate the volume of the crater
    volume_contributions_dynamic = distances[crater_mask] * pixel_areas[crater_mask]
    total_dynamic_crater_volume = np.sum(volume_contributions_dynamic)

    print(f"volume: {total_dynamic_crater_volume} m^3")

    plot(point_cloud, crater_mask, a, b, c)

if __name__ == '__main__':
    main()