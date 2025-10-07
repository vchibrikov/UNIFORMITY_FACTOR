import os
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button

# Type hinting for clarity
Point = Tuple[float, float]

class ImageProcessor:
    """
    A GUI tool to measure pixel distances between two points in a series of images.

    This class encapsulates the state and logic for loading images, handling user
    input via a Matplotlib interface, calculating distances, and saving results.
    """
    def __init__(self, image_dir: str, output_file: str):
        """
        Initializes the ImageProcessor.

        Args:
            image_dir (str): Path to the directory containing images.
            output_file (str): Path to the output Excel file for saving results.
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"The specified image directory does not exist: {image_dir}")

        self.image_dir = image_dir
        self.output_file = output_file
        
        # Discover and sort image files
        self.image_files: List[str] = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'))]
        )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in the directory: {image_dir}")

        # State variables
        self.current_image_index: int = 0
        self.points: List[Point] = []
        self.distance_results: List[List] = []
        self.define_point_active: bool = False
        
        # Image data
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None

        # Matplotlib figure and axes
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def run(self):
        """Starts the image processing workflow."""
        self._setup_ui()
        self._load_image()
        plt.show()

    def _setup_ui(self):
        """Sets up the Matplotlib UI buttons."""
        self.ax.set_title("Click 'Define Points', then select two points on the image.")
        
        ax_define = self.fig.add_axes([0.3, 0.01, 0.2, 0.05])
        btn_define = Button(ax_define, 'Define Points')
        btn_define.on_clicked(self._activate_define_mode)

        ax_save = self.fig.add_axes([0.55, 0.01, 0.2, 0.05])
        btn_save = Button(ax_save, 'Save & Next Image')
        btn_save.on_clicked(self._save_and_next)
    
    def _load_image(self):
        """Loads and displays the current image."""
        if self.current_image_index >= len(self.image_files):
            print("All images processed. Closing application.")
            plt.close(self.fig)
            return

        filename = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_dir, filename)
        
        self.original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.original_image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            self._next_image()
            return
        
        # Ensure image is in a displayable format (RGB)
        if len(self.original_image.shape) == 2:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        if self.original_image.dtype != np.uint8:
            self.original_image = self.original_image.astype(np.uint8)

        self.display_image = self.original_image.copy()
        self.points = []
        self.define_point_active = False
        self._update_display()
        self.ax.set_title(f"Processing: {filename}")

    def _update_display(self):
        """Redraws the image and any selected points on the canvas."""
        self.ax.clear()
        self.ax.imshow(self.display_image)

        # Draw the selected points
        for x, y in self.points:
            self.ax.plot(x, y, 'ro', markersize=5) # 'ro' for red circle markers

        if len(self.points) == 2:
            pt1, pt2 = self.points
            self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-') # Draw a line between points

        self.fig.canvas.draw_idle()

    def _on_click(self, event: MouseEvent):
        """Handles mouse click events on the image canvas."""
        if not self.define_point_active or event.inaxes != self.ax or len(self.points) >= 2:
            return
        
        self.points.append((event.xdata, event.ydata))
        print(f"Point {len(self.points)} selected at: ({event.xdata:.2f}, {event.ydata:.2f})")
        self._update_display()

    def _activate_define_mode(self, event: MouseEvent):
        """Activates point definition mode."""
        print("Define mode activated. Please select two points.")
        self.define_point_active = True
        self.points = []  # Reset points when starting a new definition
        self._update_display()

    def _save_and_next(self, event: MouseEvent):
        """Saves the calculated distance and loads the next image."""
        if len(self.points) != 2:
            print("Error: Please select exactly two points before saving.")
            return

        pt1, pt2 = self.points
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        
        filename = self.image_files[self.current_image_index]
        self.distance_results.append([filename, distance])
        print(f"Saved distance for {filename}: {distance:.2f} pixels.")
        
        self._save_to_excel()
        self._next_image()

    def _next_image(self):
        """Increments the image index and loads the next image."""
        self.current_image_index += 1
        self._load_image()

    def _save_to_excel(self):
        """Saves the collected distance results to an Excel file."""
        if not self.distance_results:
            return
        
        try:
            df = pd.DataFrame(self.distance_results, columns=["filename", "distance_px"])
            df.to_excel(self.output_file, index=False)
            print(f"Results successfully saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving to Excel file: {e}")

def main():
    """Main function to parse arguments and run the processor."""
    parser = argparse.ArgumentParser(description="GUI Tool to measure pixel distance in images.")
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing images."
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        required=True, 
        help="Path for the output Excel file (e.g., 'results.xlsx')."
    )
    args = parser.parse_args()

    try:
        processor = ImageProcessor(image_dir=args.input_dir, output_file=args.output_file)
        processor.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
