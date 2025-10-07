import os
import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseEvent, KeyEvent
from PIL import Image

# Define a type alias for a crop area for better readability
CropArea = Tuple[int, int, int, int]

class InteractiveCropper:
    """
    A GUI tool for interactively selecting and saving multiple cropped regions
    from a series of images.
    """
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes the InteractiveCropper.

        Args:
            input_dir (str): The directory containing images to be cropped.
            output_dir (str): The directory where cropped images will be saved.
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all supported image files
        supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files: List[str] = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)]
        )

        if not self.image_files:
            print(f"Warning: No images found in {input_dir}")
        
        # State for the current image being processed
        self.current_image_path: str = ""
        self.crop_areas: List[CropArea] = []
        self.fig: Figure
        self.ax: Axes
        
    def run(self):
        """Starts the main loop to process each image."""
        print(f"Found {len(self.image_files)} images to process.")
        for image_name in self.image_files:
            self.current_image_path = os.path.join(self.input_dir, image_name)
            self._select_areas_for_image()
            self._save_cropped_images(image_name)
        print("\nAll images have been processed. Exiting.")

    def _select_areas_for_image(self):
        """Displays an image and allows the user to select crop areas."""
        self.crop_areas = []  # Reset areas for the new image

        try:
            img = Image.open(self.current_image_path)
        except IOError:
            print(f"Error: Failed to open {self.current_image_path}. Skipping.")
            return

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.imshow(img)
        self.ax.set_title("Draw rectangles to select crop areas. Press 'Enter' to confirm.")
        self.ax.axis('off')

        # The RectangleSelector instance must be stored to keep it active
        self._selector = RectangleSelector(
            self.ax, self._onselect, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.show() # This is a blocking call until the window is closed

    def _onselect(self, eclick: MouseEvent, erelease: MouseEvent):
        """Callback for when a rectangle is selected."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        # Ensure the coordinates are in the correct order (top-left, bottom-right)
        crop_box: CropArea = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.crop_areas.append(crop_box)
        print(f"Area {len(self.crop_areas)} selected: {crop_box}")

    def _on_key_press(self, event: KeyEvent):
        """Callback for key press events."""
        if event.key == 'enter':
            print(f"Finished selecting {len(self.crop_areas)} areas for this image.")
            plt.close(self.fig)

    def _save_cropped_images(self, original_image_name: str):
        """Saves all the selected crop areas for the current image."""
        if not self.crop_areas:
            print(f"No areas were selected for {original_image_name}. Skipping save.")
            return

        try:
            with Image.open(self.current_image_path) as img:
                base_name, ext = os.path.splitext(original_image_name)
                for i, crop_area in enumerate(self.crop_areas, 1):
                    cropped_img = img.crop(crop_area)
                    
                    # Construct new filename
                    new_filename = f"{base_name}_{i}{ext}"
                    save_path = os.path.join(self.output_dir, new_filename)
                    
                    # Save with high quality
                    cropped_img.save(save_path, quality=95)
                    print(f"Saved cropped image to: {save_path}")

        except Exception as e:
            print(f"An error occurred while saving crops for {original_image_name}: {e}")

def main():
    """Parses command-line arguments and runs the image cropper."""
    parser = argparse.ArgumentParser(description="An interactive tool to crop multiple areas from images.")
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing source images."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where cropped images will be saved."
    )
    args = parser.parse_args()

    try:
        cropper = InteractiveCropper(args.input_dir, args.output_dir)
        cropper.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
