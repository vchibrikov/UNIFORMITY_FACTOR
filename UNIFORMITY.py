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
area_threshold =   # Minimum area for an object to be kept
width_step =   # Step for drawing vertical perpendiculars

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
    edges_filtered = cv2.Canny(filtered_image, 50, 150)  # Edge detection
    y_coords, x_coords = np.where(edges_filtered > 0)  # Extract edge coordinates
    return list(zip(x_coords, y_coords))

# Extract and plot filtered edges in blue
def plot_filtered_edges(ax, filtered_image):
    edges_filtered = cv2.Canny(filtered_image, 50, 150)  # Edge detection
    y_coords, x_coords = np.where(edges_filtered > 0)  # Extract edge coordinates
    ax.scatter(x_coords, y_coords, color='grey', s=2)  # Plot edges in blue

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
        ax.plot(x_smooth, y_smooth, color='grey', linewidth=2)
    else:
        ax.plot(x_values, y_values, color='grey', linewidth=2)

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
def calculate_average_angle_for_each_midpoint(min_max_y_values, n=50):
    """
    Calculate a specific tilt angle for each perpendicular based on local slope.
    """
    angles_dict = {}
    keys = list(min_max_y_values.keys())

    for i in range(len(keys)):
        x = keys[i]
        if min_max_y_values[x][0] is None or min_max_y_values[x][1] is None:
            continue  # Skip invalid points

        # Define range for slope calculation (+-n points)
        x1 = keys[max(i - n, 0)]
        x2 = keys[min(i + n, len(keys) - 1)]

        if min_max_y_values[x1][0] is None or min_max_y_values[x1][1] is None:
            continue
        if min_max_y_values[x2][0] is None or min_max_y_values[x2][1] is None:
            continue

        midpoint_1 = (x1, (min_max_y_values[x1][0] + min_max_y_values[x1][1]) / 2)
        midpoint_2 = (x2, (min_max_y_values[x2][0] + min_max_y_values[x2][1]) / 2)

        angle = calculate_angle(midpoint_1, midpoint_2) + 90
        angles_dict[x] = angle

    return angles_dict  # Dictionary of {x: angle}

# Function to calculate the length of the perpendicular line
def calculate_perpendicular_length(start, end):
    return np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

def save_figure(fig, image_name):
    output_dir = "path/to/output/image/directory"  # Change this to your desired path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    
    output_file = os.path.join(output_dir, f"{image_name.split('.')[0]}_processed.png")  
    fig.savefig(output_file, bbox_inches='tight', dpi=1200)
    print(f"Saved figure to {output_file}")

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
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='grey', linewidth=1)
    
    return perpendicular_length

def draw_tilted_perpendiculars_at_midpoints(ax, image, min_max_y_values, angles_dict, step=width_step, image_name=""):
    lengths = []
    midpoints = list(min_max_y_values.items())

    for i in range(0, len(midpoints), step):
        x, (min_y, max_y) = midpoints[i]
        if min_y is not None and max_y is not None and x in angles_dict:
            midpoint = (x, (min_y + max_y) // 2)
            angle = angles_dict[x]  # Use the specific angle for this x

            length_up = plot_tilted_perpendicular(ax, image, midpoint, angle, direction='up')
            length_down = plot_tilted_perpendicular(ax, image, midpoint, angle, direction='down')

            perpendicular_sum = length_up + length_down
            lengths.append([image_name, perpendicular_sum])

    return lengths

def save_perpendicular_lengths_to_excel(all_lengths, image_name):
    # Convert list of lengths to a DataFrame with additional columns for both perpendiculars and their sum
    df = pd.DataFrame(all_lengths, columns=['filename', 'perpendicular_length_px'])
    
    # Define the output directory and file name
    output_dir = "RESULTS/UNIFORMITY"  # You can specify the full path here if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    
    # Create the file path
    output_file = os.path.join(output_dir, f"{image_name.split('.')[0]}_UNIFORMITY.xlsx")
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Saved to {output_file}")

def update(val):
    global image, ax, fig, average_angle, image_filename

    lower_threshold = slider_lower.val
    upper_threshold = slider_upper.val

    # Process the image with the new thresholds
    _, edges, edges_closed = process_image(image, lower_threshold, upper_threshold)
    filtered_image = filter_and_visualize_objects(edges_closed, area_threshold)
    min_max_y_values = find_min_max_y(filtered_image)
    
    # Recalculate average angle based on updated midpoints
    n = 50  # You can adjust this value
    local_angle = calculate_average_angle_for_each_midpoint(min_max_y_values, n=50)
    
    # Clear the plot and update with new data
    ax.clear()
    ax.imshow(filtered_image, cmap='gray')
    plot_filtered_edges(ax, filtered_image)
    plot_midpoint_line(ax, min_max_y_values)
    
    # Draw perpendiculars and collect lengths
    lengths = draw_tilted_perpendiculars_at_midpoints(ax, filtered_image, min_max_y_values, local_angle, step=width_step, image_name=image_filename)
    
    ax.axis('off')
    fig.canvas.draw_idle()

    # **Save Figure After Updating**
    save_figure_without_sliders(fig, image_filename)  # Save the updated figure (without sliders)

    # Save lengths to Excel after sliders update
    save_perpendicular_lengths_to_excel(lengths, image_filename)

def save_figure_without_sliders(fig, image_name):
    """
    Save the figure without the sliders.
    """
    # Temporarily hide the sliders and their axes before saving
    for ax in fig.get_axes():
        if isinstance(ax, plt.Axes):
            # Check if the axis is related to a slider (you can adjust the label or properties here)
            if 'threshold' in ax.get_label().lower():  # Assuming sliders have labels with 'threshold' in them
                ax.set_visible(False)  # Hide the axes associated with the sliders
            else:
                ax.set_facecolor('white')  # Ensure no background color from other axes
    
    # Save the figure without sliders
    output_dir = "path/to/output/image/directory"  # Change this to your desired path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    
    output_file = os.path.join(output_dir, f"{image_name.split('.')[0]}_processed.png")
    
    # Save the image with a 1200 DPI resolution, excluding the slider axes
    fig.savefig(output_file, bbox_inches='tight', dpi=1200)
    print(f"Saved figure to {output_file}")

def main():
    global image, ax, slider_lower, slider_upper, fig, average_angle, image_filename
    directory_path = 'path/to/input/image/directory'
    image_files, images = load_images_from_directory(directory_path)
    
    if not images:
        print("No images found.")
        return

    # Processing each image
    for image_filename, image in zip(image_files, images):
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Initial processing (with default thresholds)
        _, edges, edges_closed = process_image(image, 0, 12)
        filtered_image = filter_and_visualize_objects(edges_closed, area_threshold)
        min_max_y_values = find_min_max_y(filtered_image)
        
        # Calculate the average angle
        n = 50
        local_angle = calculate_average_angle_for_each_midpoint(min_max_y_values, n = 50)
        
        # Display results (for the initial thresholds)
        ax.imshow(filtered_image, cmap='gray')
        plot_filtered_edges(ax, filtered_image)  # Blue edges
        plot_midpoint_line(ax, min_max_y_values)  # Red midpoint line
        
        # Draw perpendiculars (don't save yet)
        draw_tilted_perpendiculars_at_midpoints(ax, filtered_image, min_max_y_values, local_angle, step=width_step, image_name=image_filename)
        
        ax.axis('off')

        # Save initial figure and Excel file with unupdated thresholds
        save_figure_without_sliders(fig, image_filename)  # Save the initial figure without sliders
        lengths_initial = draw_tilted_perpendiculars_at_midpoints(ax, filtered_image, min_max_y_values, local_angle, step=width_step, image_name=image_filename)
        save_perpendicular_lengths_to_excel(lengths_initial, image_filename)  # Save lengths to Excel for initial thresholds
        
        # Add sliders for user interaction
        ax_slider_upper = plt.axes([0.2, 0.02, 0.65, 0.03])
        ax_slider_lower = plt.axes([0.2, 0.06, 0.65, 0.03])
        
        slider_lower = Slider(ax_slider_lower, 'Lower Threshold', 0, 25, valinit=0, valstep=1)
        slider_upper = Slider(ax_slider_upper, 'Upper Threshold', 0, 25, valinit=12, valstep=1)
        
        slider_lower.on_changed(update)  # Update figure when slider changes
        slider_upper.on_changed(update)  # Update figure when slider changes
        
        plt.subplots_adjust()
        plt.show()
    
if __name__ == '__main__':
    main()
