import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math
import pandas as pd

# Parameters (modify as needed)
area_threshold =  # Minimum area for an object to be kept
width_step = # Step for drawing  perpendiculars

# Load images
def load_images_from_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    images = [cv2.imread(os.path.join(directory_path, file), cv2.IMREAD_GRAYSCALE) for file in image_files]
    return image_files, images

# Edge detection and morphological closing
def process_image(image, lower_threshold, upper_threshold):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)
    kernel = np.ones((9, 9), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return blurred_image, edges, edges_closed

# Filter objects based on area
def filter_and_visualize_objects(edges_closed, area_threshold):
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros(edges_closed.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
    return filtered_image

# Find min and max y-values
def find_min_max_y(filtered_image):
    min_max_dict = {}
    height, width = filtered_image.shape
    for x in range(width):
        y_coords = np.where(filtered_image[:, x] == 255)[0]
        min_max_dict[x] = (np.min(y_coords), np.max(y_coords)) if len(y_coords) > 0 else (None, None)
    return min_max_dict

# Extract edges from filtered_image and collect coordinates
def collect_edge_coordinates(filtered_image):
    edges_filtered = cv2.Canny(filtered_image, 50, 150)
    y_coords, x_coords = np.where(edges_filtered > 0)
    return list(zip(x_coords, y_coords))

# Extract and plot filtered edges
def plot_filtered_edges(ax, filtered_image):
    edges_filtered = cv2.Canny(filtered_image, 50, 150)
    y_coords, x_coords = np.where(edges_filtered > 0)
    ax.scatter(x_coords, y_coords, color='blue', s=2)

# Plot smooth midpoint line
def plot_midpoint_line(ax, min_max_y_values, smooth_factor=5):
    x_values, y_values = [], []
    for x, (min_y, max_y) in min_max_y_values.items():
        if min_y is not None and max_y is not None:
            x_values.append(x)
            y_values.append((min_y + max_y) / 2)
    if len(x_values) > 3:
        spline = UnivariateSpline(x_values, y_values, s=smooth_factor)
        x_smooth = np.linspace(min(x_values), max(x_values), len(x_values) * 5)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, color='red', linewidth=2)
    else:
        ax.plot(x_values, y_values, color='red', linewidth=2)

# Collect midpoint coordinates
def collect_midpoint_coordinates(min_max_y_values):
    x_values, y_values = [], []
    for x, (min_y, max_y) in min_max_y_values.items():
        if min_y is not None and max_y is not None:
            x_values.append(x)
            y_values.append((min_y + max_y) / 2)
    return list(zip(x_values, y_values))

# Function to calculate the angle between two points
def calculate_angle(p1, p2):
    """
    Calculate the angle between the horizontal axis and the line connecting two points.
    Returns the angle in degrees.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)  # atan2 gives the angle in radians between the x-axis and the line connecting the points
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Function to calculate the average angle for 'n' previous and consecutive midpoints
def calculate_average_angle(min_max_y_values, n):
    """
    Calculate the average angle for 'n' previous and consecutive midpoints.
    'n' refers to how many previous and consecutive midpoints to consider.
    """
    angles = []
    keys = list(min_max_y_values.keys())
    
    # Loop to calculate angle between each set of 'n' previous and consecutive midpoints
    for i in range(n, len(keys) - n):
        # Skip invalid values (None for min or max)
        if (min_max_y_values[keys[i - n]][0] is None or min_max_y_values[keys[i - n]][1] is None or
            min_max_y_values[keys[i + n]][0] is None or min_max_y_values[keys[i + n]][1] is None):
            continue  # Skip this iteration if any of the values are None

        midpoint_1 = (keys[i - n], (min_max_y_values[keys[i - n]][0] + min_max_y_values[keys[i - n]][1]) / 2)
        midpoint_2 = (keys[i + n], (min_max_y_values[keys[i + n]][0] + min_max_y_values[keys[i + n]][1]) / 2)
        angle = calculate_angle(midpoint_1, midpoint_2) + 90
        angles.append(angle)
    
    # Return the average of the calculated angles
    return np.mean(angles) if angles else 0

# Function to calculate the length of the perpendicular line
def calculate_perpendicular_length(start, end):
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    
# Function to plot a tilted perpendicular line based on the average angle
def plot_tilted_perpendicular(ax, image, midpoint, angle, direction='up'):
    """
    Plot a perpendicular line tilted by the given angle.
    """
    x, y = midpoint
    step = 0
    if direction == 'up':
        step = 1
    else:
        step = -1
    
    # Calculate the direction of the perpendicular line based on the angle
    angle_rad = math.radians(angle)  # Convert angle to radians
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    # Starting point of the perpendicular line
    start_point = (x, y)

    while True:
        # Calculate the new y position based on the angle
        y_new = y + step * sin_angle
        x_new = x + step * cos_angle

        # Check if the new position is within image bounds
        if 0 <= y_new < image.shape[0] and 0 <= x_new < image.shape[1]:
            # Check if the pixel is black (0)
            if image[int(y_new), int(x_new)] == 0:
                # Stop at the first black pixel
                break
        else:
            break
        
        # Update the position for the next iteration
        y = y_new
        x = x_new

    end_point = (x, y)  # Ending point of the perpendicular line

    # Calculate the length of the perpendicular
    perpendicular_length = calculate_perpendicular_length(start_point, end_point)

    # Plot the perpendicular line
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green', linewidth=1)
    
    return perpendicular_length

def draw_tilted_perpendiculars_at_midpoints(ax, image, min_max_y_values, angle, step=width_step, image_name=""):
    lengths = []  # List to store lengths of perpendiculars and their sums
    midpoints = list(min_max_y_values.items())
    
    for i in range(0, len(midpoints), step):
        x, (min_y, max_y) = midpoints[i]
        if min_y is not None and max_y is not None:
            midpoint = (x, (min_y + max_y) // 2)
            
            # Draw a tilted perpendicular from each midpoint and collect lengths
            length_up = plot_tilted_perpendicular(ax, image, midpoint, angle, direction='up')
            length_down = plot_tilted_perpendicular(ax, image, midpoint, angle, direction='down')
            
            # Calculate the sum of both perpendicular lengths
            perpendicular_sum = length_up + length_down
            
            # Store the image name, perpendiculars and their sum
            lengths.append([image_name, perpendicular_sum])
    
    return lengths

def save_perpendicular_lengths_to_excel(all_lengths, image_name):
    # Convert list of lengths to a DataFrame with additional columns for both perpendiculars and their sum
    df = pd.DataFrame(all_lengths, columns=['filename', 'perpendicular_length_px'])
    
    # Define the output directory and file name
    output_dir = "RESULTS"  # You can specify the full path here if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    
    # Create the file path
    output_file = os.path.join(output_dir, f"{image_name.split('.')[0]}_UNIFORMITY.xlsx")
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Saved to {output_file}")

# Update function for slider interaction
def update(val):
    lower_threshold = slider_lower.val
    upper_threshold = slider_upper.val
    _, edges, edges_closed = process_image(image, lower_threshold, upper_threshold)
    filtered_image = filter_and_visualize_objects(edges_closed, area_threshold)
    min_max_y_values = find_min_max_y(filtered_image)
    
    # Recalculate average angle based on updated midpoints
    n = 50  # You can adjust this value
    average_angle = calculate_average_angle(min_max_y_values, n)
    
    # Update the plot with the new image and updated perpendiculars
    ax.clear()
    ax.imshow(filtered_image, cmap='gray')
    plot_filtered_edges(ax, filtered_image)  # Blue edges from filtered_image
    plot_midpoint_line(ax, min_max_y_values)  # Red midpoint line
    draw_tilted_perpendiculars_at_midpoints(ax, filtered_image, min_max_y_values, average_angle, step=width_step)
    ax.axis('off')
    fig.canvas.draw_idle()

def main():
    global image, ax, slider_lower, slider_upper, fig, average_angle
    directory_path = '' # Specify the directory path here
    image_files, images = load_images_from_directory(directory_path)
    
    if not images:
        print("No images found.")
        return

    # Processing each image
    for image_filename, image in zip(image_files, images):
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Initial processing
        _, edges, edges_closed = process_image(image, 0, 12)
        filtered_image = filter_and_visualize_objects(edges_closed, area_threshold)
        min_max_y_values = find_min_max_y(filtered_image)
        
        # Calculate the average angle for n midpoints (e.g., n = 50)
        n = 50
        average_angle = calculate_average_angle(min_max_y_values, n)
        
        # Collect coordinates
        edge_coords = collect_edge_coordinates(filtered_image)
        midpoint_coords = collect_midpoint_coordinates(min_max_y_values)
        
        # Display results
        ax.imshow(filtered_image, cmap='gray')
        plot_filtered_edges(ax, filtered_image)  # Blue edges from filtered_image
        plot_midpoint_line(ax, min_max_y_values)  # Red midpoint line
        
        # Draw tilted perpendiculars at every 1000th midpoint based on the average angle
        lengths = draw_tilted_perpendiculars_at_midpoints(ax, filtered_image, min_max_y_values, average_angle, step=width_step, image_name=image_filename)
        
        ax.axis('off')

        # Add sliders
        ax_slider_upper = plt.axes([0.2, 0.02, 0.65, 0.03])
        ax_slider_lower = plt.axes([0.2, 0.06, 0.65, 0.03])
        
        slider_lower = Slider(ax_slider_lower, 'Lower Threshold', 0, 25, valinit=0, valstep=1)
        slider_upper = Slider(ax_slider_upper, 'Upper Threshold', 0, 25, valinit=14, valstep=1)
        
        slider_lower.on_changed(update)
        slider_upper.on_changed(update)
        
        plt.subplots_adjust()
        plt.show()

        # Save lengths to Excel with the image name
        save_perpendicular_lengths_to_excel(lengths, image_filename)
    
if __name__ == '__main__':
    main()