# PB-Lite Algorithm for Edge Detection

![PB-Lite Banner](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/pb_lite_banner.png)

## Overview

The PB-Lite (Probability of Boundary - Lite) algorithm is an efficient edge detection framework inspired by advanced contour detection techniques. It combines multiple gradient-based and clustering methods to generate accurate edge-detection outputs for images. This project implements the PB-Lite algorithm using Python and evaluates its performance on 10 images from the BSDS500 dataset.

The output of this phase is a set of edge-detected images, emphasizing object boundaries, along with visualizations of intermediate filter banks and gradient maps.

## Features

- **Filter Banks**: Implements Leung-Malik (LM), Difference of Gaussian (DoG), and Gabor filter banks.
- **Map Generation**: Produces texture, brightness, and color maps using K-means clustering.
- **Gradient Computation**: Computes gradients using Chi-square distance and half-disc filters.
- **Edge Detection**: Combines multiple gradient responses with baseline edge-detection methods (Canny, Sobel).
- **Modular Design**: Allows flexibility for experimentation and fine-tuning.

## Methodology

1. **Input Preprocessing**:
   - Load images from the BSDS500 dataset.
   - Convert images to required formats (grayscale or RGB).

2. **Filter Bank Generation**:
   - Leung-Malik filters for textural information.
   - Difference of Gaussian filters for multi-scale edge detection.
   - Gabor filters for orientation-specific features.

3. **Map Creation**:
   - Texture maps using filter bank responses and K-means clustering.
   - Brightness and color maps using clustering on pixel intensities.

4. **Gradient Computation**:
   - Chi-square gradients using half-disc masks.

5. **Edge Detection**:
   - Combine texture, brightness, and color gradients with Sobel and Canny baselines.
   - Generate PB-Lite edge-detected outputs.

6. **Visualization**:
   - Save and display intermediate maps, gradients, and final outputs.

## Installation

### Prerequisites

Ensure you have Python 3.8 or later installed. Install additional dependencies listed in the `requirements.txt` file.

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/pb-lite-edge-detection.git
   cd pb-lite-edge-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the folder structure for results:
   ```bash
   python folders.py
   ```

## Usage

### Running the Code

Run the main script to process all 10 images in the dataset and generate outputs:
```bash
python pb_lite.py
```

Optional argument:
- `--Maps_flag`: If set to `False`, precomputed maps will be loaded instead of generating new ones.

Example:
```bash
python pb_lite.py --Maps_flag=False
```

### Outputs

- Filter banks visualizations: Saved in `results/filterbanks/`.
- Maps (texture, brightness, color): Saved in `results/maps/`.
- Gradients and final edge-detected images: Saved in `results/edges/imgX/` (where `X` is the image number).

## Results

### Example Outputs

Below are sample outputs from the PB-Lite algorithm:

#### Filter Banks
![LM Filter Bank](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/lm_filter_bank.png)
![DoG Filter Bank](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/dog_filter_bank.png)
![Gabor Filter Bank](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/gabor_filter_bank.png)

#### Edge Detection Results
![Edge Detection Result 1](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/edge_result_1.png)
![Edge Detection Result 2](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/edge_result_2.png)

## Fine-Tuning for Better Performance

1. **Adjust Filter Parameters**:
   - Modify kernel sizes, orientations, and scales for LM, DoG, and Gabor filters in the script.

2. **Clustering Optimization**:
   - Experiment with different values of `num_clusters` in texture, brightness, and color map generation.

3. **Gradient Weights**:
   - Adjust the weights `alpha`, `beta`, and `gamma` in the edge combination step for optimal blending.

4. **Dataset**:
   - Test the algorithm on additional datasets to generalize performance.

5. **Baseline Blending**:
   - Experiment with different combinations of Sobel and Canny baselines.

---

For more details or to contribute, please contact [your-email@example.com].

