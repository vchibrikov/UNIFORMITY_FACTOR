# UNIFORMITY.py
- Current repository provides an image processing script that performs edge detection, object filtering, and calculates the lengths of perpendiculars drawn from midpoints of object boundaries in images. The program can also plot the filtered objects, midpoint lines, and tilted perpendiculars, with interactive sliders to adjust threshold values for edge detection.
- Repository is created for the project entitled "Printing of 3D biomaterialsinspired by plant cell wall", supported by the National Science Centre, Poland (grant nr - 2023/49/B/NZ9/02979).
- Research methodology is an automative approach used in paper: Merli M, Sardelli L, Baranzini N, Grimaldi A, Jacchetti E, Raimondi MT, Briatico-Vangosa F, Petrini P and Tunesi M (2022), Pectin-based bioinks for 3D models of neural tissue produced by a pH-controlled kinetics. Front. Bioeng. Biotechnol. 10:1032542. doi: 10.3389/fbioe.2022.1032542

## Requirements

The following Python libraries are required:

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- SciPy
- pandas

You can install the required libraries by running:

## Description
### Main Functionality:
- Edge detection and morphological processing: the script performs Gaussian blur and edge detection on the input images. The edges are processed using morphological closing to enhance the object boundaries (Fig. 1-3).
- Object filtering: the code filters objects based on their area and visualizes the result by drawing contours around objects that exceed a defined area threshold.

![fig_1](https://github.com/user-attachments/assets/a174ff84-c8fe-4a8b-a473-d2f78ad16cd8)
Figure 1. Raw data image.

![fig_3](https://github.com/user-attachments/assets/53f96bfa-582f-44f0-9302-d5b9133dfd0e)
Figure 2. Object boundaries detected.

![fig_2](https://github.com/user-attachments/assets/71189419-e2e7-4b68-b09d-a5fd3b43a2ff)
Figure 3. Morphologically closed object.

- Midpoint line calculation: the midpoints between the upper and lower boundaries of each column in the filtered image are calculated. A smooth spline is fitted through these midpoints to create a midpoint line (Fig. 3).

![fig_4](https://github.com/user-attachments/assets/fa9201f8-c958-44fd-9eaa-5b85db803697)
Figure 3. Object with a midpoint line.

- Perpendiculars and angles: tilted perpendicular lines are drawn from each midpoint. The direction and length of the perpendiculars are calculated based on the average angle of previous midpoints (Fig.4). The lengths of these perpendiculars are saved in an Excel file for analysis.

![fig_5](https://github.com/user-attachments/assets/770017b2-221f-41d5-8ba5-7ae9db60d2c3)
Figure 4. Perpendiculars detected.

- Interactive sliders: two sliders allow users to adjust the lower and upper thresholds for edge detection in real time. The plot updates interactively as the sliders are adjusted

- Saving results: the perpendicular lengths and other details are saved to an Excel file for each image processed
- Files processed: the script processes all images in a specified directory (directory_path). The images should be in .jpeg, .jpg, or .png format

### Parameters
- area_threshold: Minimum area for an object to be retained
- width_step: Step size for drawing perpendiculars at midpoints
- lower_threshold: The lower threshold value for edge detection
- upper_threshold: The upper threshold value for edge detection

### Usage
- Set up the directory path
- Run the script:
- Use the sliders to adjust the lower_threshold and upper_threshold for edge detection

### Output:
- The processed results will be shown in the plot
- Perpendicular lengths will be saved in an Excel file in the RESULTS folder

### Notes
- Ensure that the images you are using are of good quality with clear object boundaries for better edge detection results.
- The program can be adapted to process a larger variety of images by adjusting the parameters.

### License
This code is licensed under the MIT License. See the LICENSE file for more details.
