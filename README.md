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
- Edge detection and morphological processing: the script performs Gaussian blur and edge detection on the input images. The edges are processed using morphological closing to enhance the object boundaries (Fig. 1-2).
- Object filtering: the code filters objects based on their area and visualizes the result by drawing contours around objects that exceed a defined area threshold.
![Figure_1](https://github.com/user-attachments/assets/00195e51-0f57-4c7b-aa77-9ad16cce6815)
Figure 1. Raw data image.
![Figure_2](https://github.com/user-attachments/assets/73b9311d-7291-4bcf-beaa-e9cf64aa523d)
Figure 2. Morphologically closed object.

- Midpoint line calculation: the midpoints between the upper and lower boundaries of each column in the filtered image are calculated. A smooth spline is fitted through these midpoints to create a red midpoint line (Fig. 3).
![Figure_3](https://github.com/user-attachments/assets/327d5cc2-9993-4d2d-a30f-6f32e3c64b66)
Figure 3. Object with a midpoint line.

- Perpendiculars and angles: tilted perpendicular lines are drawn from each midpoint. The direction and length of the perpendiculars are calculated based on the average angle of previous midpoints (Fig.4). The lengths of these perpendiculars are saved in an Excel file for analysis.
![Figure_5](https://github.com/user-attachments/assets/62339857-0b02-4d27-893e-ecbf2f62b17f)
Figure 3. Perpendiculars detected.

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
