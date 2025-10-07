import os
import math
import argparse
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import UnivariateSpline

# --- Type Hinting for Clarity ---
Image = np.ndarray
Point = Tuple[float, float]
MinMaxY = Dict[int, Tuple[Optional[int], Optional[int]]]
Angles = Dict[int, float]

class ObjectWidthAnalyzer:
    """
    An interactive tool to analyze the width of elongated objects in images.

    This class provides a GUI to:
    1. Load images and perform Canny edge detection with adjustable thresholds.
    2. Filter objects by area to isolate the main object of interest.
    3. Calculate the object's centerline and draw perpendicular lines along it.
    4. Measure the length of these perpendiculars to determine local width.
    5. Save the annotated image and width data to an Excel file.
    """
    def __init__(self, input_dir: str, output_dir: str, params: dict):
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M'))
        self.img_output_dir = os.path.join(self.output_dir, 'images')
        self.excel_output_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.excel_output_dir, exist_ok=True)

        self.params = params
        self.image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

        # --- State variables for the current image ---
        self.current_image: Optional[Image] = None
        self.current_filename: str = ""
        self.fig = None
        self.ax = None
        self.ax_slider_lower = None
        self.ax_slider_upper = None
        self.ax_button = None
        self.slider_lower = None
        self.slider_upper = None
        self.button = None
        self.analysis_results: dict = {}

    def run(self):
        """Processes each image in the input directory interactively."""
        print(f"Found {len(self.image_files)} images. Starting analysis...")
        for filename in self.image_files:
            self.current_filename = filename
            image_path = os.path.join(self.input_dir, filename)
            self.current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self.current_image is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue
            
            self._process_image_interactively()
        print("\nAll images processed. Results saved in:", self.output_dir)

    def _process_image_interactively(self):
        """Sets up the Matplotlib window with sliders and a button for one image."""
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.2) # Make space for widgets

        # --- Create widgets ---
        self.ax_slider_lower = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_slider_upper = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.ax_button = plt.axes([0.8, 0.9, 0.1, 0.04]) # Position button in top right

        self.slider_lower = Slider(self.ax_slider_lower, 'Lower Threshold', 0, 255, valinit=self.params['canny_lower'], valstep=1)
        self.slider_upper = Slider(self.ax_slider_upper, 'Upper Threshold', 0, 255, valinit=self.params['canny_upper'], valstep=1)
        self.button = Button(self.ax_button, 'Save & Next')

        self.slider_lower.on_changed(self._update_plot)
        self.slider_upper.on_changed(self._update_plot)
        self.button.on_clicked(self._save_and_close)

        self.ax.set_title(f"Analyzing: {self.current_filename}")
        self._update_plot() # Initial plot
        plt.show() # Blocks until window is closed

    def _update_plot(self, val=None):
        """Performs the full analysis pipeline and redraws the plot."""
        lower_thresh = self.slider_lower.val
        upper_thresh = self.slider_upper.val

        # 1. Edge detection and closing
        blurred = cv2.GaussianBlur(self.current_image, (5, 5), 0)
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
        kernel = np.ones((9, 9), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 2. Filter objects by area
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_image = np.zeros_like(edges_closed)
        for c in contours:
            if cv2.contourArea(c) > self.params['area_threshold']:
                cv2.drawContours(filtered_image, [c], -1, 255, thickness=cv2.FILLED)

        # 3. Find object boundaries and centerline
        min_max_y = self._find_min_max_y(filtered_image)
        
        # 4. Calculate local angles for perpendiculars
        angles = self._calculate_local_angles(min_max_y)

        # --- Store results for saving later ---
        self.analysis_results = {
            "filtered_image": filtered_image,
            "min_max_y": min_max_y,
            "angles": angles
        }

        # --- Redraw the plot ---
        self.ax.clear()
        self.ax.imshow(self.current_image, cmap='gray')
        
        # Plot Edges
        edge_y, edge_x = np.where(cv2.Canny(filtered_image, 50, 150) > 0)
        self.ax.scatter(edge_x, edge_y, color='cyan', s=1, alpha=0.5)

        # Plot Midpoint Line
        self._plot_midpoint_line(self.ax, min_max_y)
        
        # Plot Perpendiculars
        self._draw_perpendiculars(self.ax, filtered_image, min_max_y, angles)
        
        self.ax.axis('off')
        self.ax.set_title(f"Analyzing: {self.current_filename}")
        self.fig.canvas.draw_idle()

    def _save_and_close(self, event):
        """Saves the current analysis results and closes the window."""
        print(f"Saving results for {self.current_filename}...")
        
        # --- Save the figure ---
        # Hide widgets before saving for a clean image
        self.ax_slider_lower.set_visible(False)
        self.ax_slider_upper.set_visible(False)
        self.ax_button.set_visible(False)
        
        img_savename = f"{os.path.splitext(self.current_filename)[0]}_analyzed.png"
        img_savepath = os.path.join(self.img_output_dir, img_savename)
        self.fig.savefig(img_savepath, dpi=self.params['dpi'], bbox_inches='tight', pad_inches=0.1)
        print(f"Saved image to {img_savepath}")
        
        # --- Save the Excel data ---
        lengths = self._get_perpendicular_lengths(
            self.analysis_results["filtered_image"],
            self.analysis_results["min_max_y"],
            self.analysis_results["angles"]
        )
        df = pd.DataFrame(lengths, columns=['filename', 'width_px'])
        excel_savename = f"{os.path.splitext(self.current_filename)[0]}_width_data.xlsx"
        excel_savepath = os.path.join(self.excel_output_dir, excel_savename)
        df.to_excel(excel_savepath, index=False)
        print(f"Saved data to {excel_savepath}")

        plt.close(self.fig)

    def _find_min_max_y(self, image: Image) -> MinMaxY:
        """Finds the min and max y-coordinates for each x-column."""
        min_max = {}
        for x in range(image.shape[1]):
            y_coords = np.where(image[:, x] == 255)[0]
            if len(y_coords) > 0:
                min_max[x] = (np.min(y_coords), np.max(y_coords))
            else:
                min_max[x] = (None, None)
        return min_max

    def _calculate_local_angles(self, min_max_y: MinMaxY) -> Angles:
        """Calculates the local slope angle for each point on the centerline."""
        angles = {}
        x_coords = sorted([x for x, v in min_max_y.items() if v[0] is not None])
        n = self.params['angle_neighbors']

        for i, x in enumerate(x_coords):
            # Define neighborhood for slope calculation
            x1_idx = max(i - n, 0)
            x2_idx = min(i + n, len(x_coords) - 1)
            x1, x2 = x_coords[x1_idx], x_coords[x2_idx]

            midpoint1_y = (min_max_y[x1][0] + min_max_y[x1][1]) / 2
            midpoint2_y = (min_max_y[x2][0] + min_max_y[x2][1]) / 2

            dx = x2 - x1
            dy = midpoint2_y - midpoint1_y
            
            # Angle of the tangent line + 90 degrees for the normal
            tangent_angle_rad = math.atan2(dy, dx)
            normal_angle_deg = math.degrees(tangent_angle_rad) + 90
            angles[x] = normal_angle_deg
        return angles

    @staticmethod
    def _plot_midpoint_line(ax, min_max_y: MinMaxY, smooth_factor=5):
        """Plots a smoothed line through the center of the object."""
        x_vals, y_vals = [], []
        for x, (min_y, max_y) in min_max_y.items():
            if min_y is not None:
                x_vals.append(x)
                y_vals.append((min_y + max_y) / 2)
        
        if len(x_vals) > 3:
            spline = UnivariateSpline(x_vals, y_vals, s=smooth_factor)
            x_smooth = np.linspace(min(x_vals), max(x_vals), len(x_vals) * 5)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color='red', linewidth=1.5, label='Centerline')
        elif x_vals:
            ax.plot(x_vals, y_vals, color='red', linewidth=1.5, label='Centerline')

    def _draw_perpendiculars(self, ax, image: Image, min_max_y: MinMaxY, angles: Angles):
        """Draws the perpendicular lines on the plot."""
        step = self.params['width_step']
        x_coords = sorted(angles.keys())
        for x in x_coords[::step]:
            min_y, max_y = min_max_y[x]
            midpoint = (x, (min_y + max_y) / 2)
            angle = angles[x]

            for direction in ['up', 'down']:
                end_point = self._trace_perpendicular(image, midpoint, angle, direction)
                ax.plot([midpoint[0], end_point[0]], [midpoint[1], end_point[1]], color='yellow', linewidth=0.5)

    def _get_perpendicular_lengths(self, image: Image, min_max_y: MinMaxY, angles: Angles) -> List:
        """Calculates and returns the lengths of all perpendiculars."""
        lengths = []
        step = self.params['width_step']
        x_coords = sorted(angles.keys())
        for x in x_coords[::step]:
            min_y, max_y = min_max_y[x]
            midpoint = (x, (min_y + max_y) / 2)
            angle = angles[x]
            
            p_up = self._trace_perpendicular(image, midpoint, angle, 'up')
            p_down = self._trace_perpendicular(image, midpoint, angle, 'down')
            
            len_up = np.linalg.norm(np.array(midpoint) - np.array(p_up))
            len_down = np.linalg.norm(np.array(midpoint) - np.array(p_down))
            
            lengths.append([self.current_filename, len_up + len_down])
        return lengths

    @staticmethod
    def _trace_perpendicular(image: Image, start: Point, angle_deg: float, direction: str) -> Point:
        """Traces a line from a start point until it hits a black pixel."""
        angle_rad = math.radians(angle_deg)
        vx, vy = math.cos(angle_rad), math.sin(angle_rad) # Direction vector
        
        step_multiplier = 1.0 if direction == 'up' else -1.0
        x, y = start

        while True:
            x_new, y_new = x + step_multiplier * vx, y + step_multiplier * vy
            
            # Check image bounds
            if not (0 <= y_new < image.shape[0] and 0 <= x_new < image.shape[1]):
                break
            # Check if pixel is part of the object (non-zero)
            if image[int(y_new), int(x_new)] == 0:
                break
            x, y = x_new, y_new
            
        return (x, y)


def main():
    parser = argparse.ArgumentParser(description="Interactively analyze object width in images.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory with source images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save results.")
    parser.add_argument("--area_thresh", type=int, default=1000, help="Minimum contour area to keep an object.")
    parser.add_argument("--width_step", type=int, default=10, help="Step size (in pixels) for drawing width lines.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved images.")
    parser.add_argument("--angle_neighbors", type=int, default=50, help="Number of neighbors for local angle calculation.")
    parser.add_argument("--canny_lower", type=int, default=0, help="Initial lower Canny threshold.")
    parser.add_argument("--canny_upper", type=int, default=12, help="Initial upper Canny threshold.")
    args = parser.parse_args()

    params = {
        'area_threshold': args.area_thresh,
        'width_step': args.width_step,
        'dpi': args.dpi,
        'angle_neighbors': args.angle_neighbors,
        'canny_lower': args.canny_lower,
        'canny_upper': args.canny_upper
    }

    try:
        analyzer = ObjectWidthAnalyzer(args.input_dir, args.output_dir, params)
        analyzer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
