import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import numpy as np

# Function to handle cropping and saving images
def crop_and_save(image_path, save_directory, crop_area, image_name, counter):
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Crop the image
    cropped_img = img.crop(crop_area)
    
    # Save the cropped image with the original name and a suffix (_1, _2, etc.)
    base_name, ext = os.path.splitext(image_name)
    new_image_path = os.path.join(save_directory, f"{base_name}_{counter}{ext}")
    
    # Save the cropped image with a higher quality (quality=100 for maximum)
    cropped_img.save(new_image_path, quality=100)
    print(f"Saved cropped image as: {new_image_path}")

# Callback function to store the selected area coordinates
def onselect(eclick, erelease):
    global crop_areas
    crop_area = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
    crop_areas.append(crop_area)
    print(f"Selected crop area: {crop_area}")

# Function to select rectangle area from the image using matplotlib
def select_crop_area(image_path):
    global crop_areas
    crop_areas = []

    # Load the image using matplotlib
    try:
        img = plt.imread(image_path)
        print(f"Image {image_path} loaded successfully!")
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return
    
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Create a rectangle selector
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],
                                      minspanx=5, minspany=5, spancoords='pixels',
                                      interactive=True)
    
    # Set the figure key press handler
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    plt.title("Select the areas to crop. Press Enter when done.")
    plt.show(block=True)

# Custom key press event handler to close the plot on Enter key
def on_key_press(event):
    if event.key == 'enter':
        print("Enter key pressed. Closing the plot and saving selected areas.")
        plt.close()

# Main function to read all images from a directory and allow cropping
def process_images(input_directory, output_directory):
    # Print the directory being checked
    
    # Get all files in the directory and print them
    all_files = os.listdir(input_directory)
    
    # Get all image files from the input directory with matching extensions
    image_files = [f for f in all_files if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    
    for image_name in image_files:
        image_path = os.path.join(input_directory, image_name)
        
        # Let the user select multiple areas
        select_crop_area(image_path)
        
        # Wait for user to finish selecting and provide the coordinates
        if 'crop_areas' in globals() and crop_areas:
            # Iterate over the selected crop areas and save each cropped image
            for i, crop_area in enumerate(crop_areas, 1):
                crop_and_save(image_path, output_directory, crop_area, image_name, i)
            del globals()['crop_areas']
        else:
            print(f"No areas selected for image: {image_name}")

# Example usage
input_directory = 'path/to/input/directory'
output_directory = 'path/to/output/directory'

process_images(input_directory, output_directory)
