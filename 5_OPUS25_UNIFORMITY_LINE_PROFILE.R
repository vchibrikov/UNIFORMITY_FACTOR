# -----------------------------------------------------------------------------
#
# Script: Line Width Distribution Analysis
# Author: Vadym Chibrikov
# Date: 2025-10-07
#
# Description: This script visualizes how the width of 3D printed lines
#              changes along their length. It processes data from a Python
#              image analysis tool, normalizes multiple line replicates to a
#              common percentage-based axis, and plots the mean width profile
#              with a standard deviation ribbon.
#
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SECTION 1: SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# --- 1.1: Load Required Libraries ---
# Using pacman to install and load packages in one step
if (!require("pacman")) install.packages("pacman")
pacman::p_load('tidyverse', 'readxl', 'Cairo', 'bbplot')

# --- 1.2: Configuration ---
# EDIT THESE VARIABLES TO MATCH YOUR PROJECT
#
# Input file paths
raw_data_path <- "./raw/data/path.xlsx"
scale_data_path <- "./scale/data/path.xlsx"

# Output directory for plots
output_dir <- "./output/data/directory/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Select the experimental group to analyze ---
# Provide a title for the plot and the list of filenames for that group.
# Only one group should be uncommented at a time.
plot_title <- "TITLE"
selected_files <- c("SELECTED_FILE_1", "SELECTED_FILE_2", "SELECTED_FILE_3")


# --- 1.3: Plotting Parameters ---
axis.text.font.size <- 16
axis.title.font.size <- 18


# -----------------------------------------------------------------------------
# SECTION 2: DATA LOADING AND PREPARATION
# -----------------------------------------------------------------------------
# This section loads and prepares the data. This is done only once.

# --- 2.1: Load Data and Convert Pixels to Millimeters ---
raw_data <- read_excel(raw_data_path)
scale_data <- read_excel(scale_data_path)

full_data <- raw_data %>%
  mutate(
    date = str_extract(filename, "\\d{4}_\\d{2}_\\d{2}"),
    filename_base = str_remove(filename, "\\.png$")
  ) %>%
  left_join(
    scale_data %>%
      mutate(date = str_extract(filename, "\\d{4}_\\d{2}_\\d{2}")) %>%
      select(date, distance_10_mm_px),
    by = "date"
  ) %>%
  filter(!is.na(perpendicular_length_px), !is.na(distance_10_mm_px)) %>%
  mutate(
    perpendicular_length_mm = as.numeric(perpendicular_length_px) / (as.numeric(distance_10_mm_px) / 10)
  ) %>%
  select(filename = filename_base, perpendicular_length_mm)


# -----------------------------------------------------------------------------
# SECTION 3: CORE ANALYSIS FUNCTION
# -----------------------------------------------------------------------------
# This function contains the logic for resampling and summarizing the data.

#' Calculate Width Distribution Profile
#'
#' @param data A dataframe containing width measurements.
#' @param files_to_analyze A character vector of filenames to include.
#' @param normalize A boolean indicating whether to normalize width data.
#' @return A summarized tibble with mean and SD of width at each percentage point.
calculate_width_distribution <- function(data, files_to_analyze, normalize = FALSE) {

  # --- Step 1: Filter data and calculate trim metadata ---
  # This assumes 50px were trimmed from each end of the line during image analysis.
  data_filtered <- data %>%
    filter(filename %in% files_to_analyze)

  if (normalize) {
    data_filtered <- data_filtered %>%
      group_by(filename) %>%
      mutate(
        mean_width = mean(perpendicular_length_mm, na.rm = TRUE),
        value = perpendicular_length_mm / mean_width # Normalize
      ) %>%
      ungroup()
  } else {
    data_filtered <- data_filtered %>%
      mutate(value = perpendicular_length_mm)
  }

  data_with_meta <- data_filtered %>%
    group_by(filename) %>%
    mutate(
      points_in_line = n(),
      original_length_px = points_in_line + 100, # Add back trimmed 50px from each side
      start_percent = (50 / original_length_px) * 100,
      end_percent = ( (points_in_line + 50) / original_length_px) * 100
    ) %>%
    ungroup()

  # --- Step 2: Create a common axis and resample data ---
  # Create a shared x-axis from 0% to 100% to compare lines of different lengths.
  global_percent_axis <- seq(
    min(data_with_meta$start_percent),
    max(data_with_meta$end_percent),
    length.out = 200 # Resample to 200 points
  )

  resampled_data <- data_with_meta %>%
    group_by(filename) %>%
    # Resample each line's data onto the common global_percent_axis
    summarise(
      percent = global_percent_axis,
      resampled_value = approx(
        x = seq(first(start_percent), first(end_percent), length.out = first(points_in_line)),
        y = value,
        xout = global_percent_axis
      )$y,
      .groups = "drop"
    )

  # --- Step 3: Calculate final mean and standard deviation ---
  summary_data <- resampled_data %>%
    group_by(percent) %>%
    summarise(
      mean_value = mean(resampled_value, na.rm = TRUE),
      sd_value = sd(resampled_value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    na.omit()

  return(summary_data)
}


# -----------------------------------------------------------------------------
# SECTION 4: PLOTTING FUNCTION
# -----------------------------------------------------------------------------

#' Plot Width Distribution Profile
#'
#' @param summary_data The summarized data from calculate_width_distribution.
#' @param plot_title The main title for the plot.
#' @param y_label The label for the y-axis.
#' @param y_limits A numeric vector for y-axis limits, e.g., c(0, 5).
#' @param y_breaks A numeric value for y-axis break intervals, e.g., 1.
#' @param hline_intercept The position for a dashed horizontal line.
#' @return A ggplot object.
plot_distribution <- function(summary_data, plot_title, y_label, y_limits, y_breaks, hline_intercept) {
  ggplot(summary_data, aes(x = percent, y = mean_value)) +
    geom_hline(yintercept = hline_intercept, linetype = "dashed", color = "#333333", linewidth = 0.5) +
    geom_line(color = "black", linewidth = 1) +
    geom_ribbon(aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value), fill = "grey50", alpha = 0.3) +
    scale_y_continuous(limits = y_limits, breaks = seq(y_limits[1], y_limits[2], by = y_breaks)) +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
    labs(x = "Line length (%)", y = y_label, title = plot_title) +
    bbc_style() +
    theme(
      aspect.ratio = 0.75,
      plot.title = element_text(size = axis.title.font.size, hjust = 0.5),
      axis.title = element_text(size = axis.title.font.size, face = "bold"),
      axis.text = element_text(size = axis.text.font.size)
    )
}


# -----------------------------------------------------------------------------
# SECTION 5: EXECUTE ANALYSIS AND SAVE PLOTS
# -----------------------------------------------------------------------------

# --- 5.1: Absolute Line Width Distribution ---
summary_absolute <- calculate_width_distribution(full_data, selected_files, normalize = FALSE)
plot_absolute <- plot_distribution(
  summary_data = summary_absolute,
  plot_title = plot_title,
  y_label = "Line width (mm)",
  y_limits = c(0, 5),
  y_breaks = 1,
  hline_intercept = 0.41 # Example target width
)
ggsave(
  filename = file.path(output_dir, paste0("DISTRIBUTION_ABSOLUTE_", plot_title, ".jpeg")),
  plot = plot_absolute,
  width = 18, height = 12, units = "cm", dpi = 600, device = "cairo_jpeg"
)

# --- 5.2: Normalized Line Width Distribution ---
summary_normalized <- calculate_width_distribution(full_data, selected_files, normalize = TRUE)
plot_normalized <- plot_distribution(
  summary_data = summary_normalized,
  plot_title = plot_title,
  y_label = "Normalized line width (a.u.)",
  y_limits = c(0, 2),
  y_breaks = 0.5,
  hline_intercept = 1
)
ggsave(
  filename = file.path(output_dir, paste0("DISTRIBUTION_NORMALIZED_", plot_title, ".jpeg")),
  plot = plot_normalized,
  width = 18, height = 12, units = "cm", dpi = 600, device = "cairo_jpeg"
)

print(paste("Distribution plots for", plot_title, "saved to:", output_dir))
