# PB-Lite Algorithm for Edge Detection & Deep Learning Approaches for Image Clasification

![PB-Lite Banner](https://raw.githubusercontent.com/your-repo/pb-lite-edge-detection/main/assets/images/pb_lite_banner.png)

## Overview

The **PB-Lite (Probability of Boundary - Lite)** algorithm is an efficient edge detection framework inspired by advanced contour detection techniques. This framework is a simplified version of a recent paper '[Contour Detection and Hierarchical Image Segmentation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf)'. It combines multiple gradient-based and clustering methods to generate accurate edge-detection outputs for images. This simplified version finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image). The output of this edge detection framework is a per-pixel probability of boundary. 

We also explore multiple **Deep learning(DL)** architectures--**ResNet, ResNeXt, DenseNet** and a **Custom neural network** to enhance image classification performance. The goal is to evaluate and compare their performance in terms of accuracy, efficiency, and scalability. The project leverages **TensorFlow v1** for training and testing the models, using a custom dataset for classification.

## Features

1. **Pb-lite Edge Detection:**
    - **Filter Banks**: Create three sets of filter banks for filtering images. (Leung-Malik (LM), Difference of Gaussian (DoG), and Gabor filter banks.)
    - **Edge Detection**: Combines multiple gradient responses from texture, brightness and color to compute probability of edges with baseline edge-detection methods (Canny, Sobel).
    - **Modular Design**: Allows flexibility for experimentation and fine-tuning.

2. **Imge Classification:**
    - **Deep Learning(DL) models**: training and testing CNN, custom CNN(CNN_BN), ResNet, ResNext and Densenet for image classification.
    - **TensorFlow v1 Compatibility**: Uses TensorFlow v1 for model training, ensuring backward compatibility.
    - **Modular Design**: Each model implementation is modular and can be trained/tested independently.
    - **Performance Metrics**: Tracks training accuracy, testing accuracy, and loss.
    - **Fine-Tuning Support**: Easily fine-tune the models for better performance.

## Methodology

### Pb-lite Edge Detection

1. **Filter Bank Generation**:
    Generating filter banks for building low level features to measure texture properties and to aggregate regional texture and brightness distributions. 
   - Leung-Malik filters for textural information.
   - Difference of Gaussian filters for multi-scale edge detection.
   - Gabor filters for orientation-specific features.

3. **Map Creation**:
   - Texture maps using filter bank responses and K-means clustering.
   - Brightness and color maps using clustering on pixel intensities.

4. **Gradient Computation**:
   - Chi-square gradients using half-disc masks to highlight edges.

5. **Edge Detection**:
   - Combine texture, brightness, and color gradients with Sobel and Canny baselines To generate PB-Lite edge-detected outputs using probabilistic methods.

Read More in detail here: [CMSC-733, Homework0:Alohomora](https://cmsc733.github.io/2022/hw/hw0/#starter)

### Deep Learning techniques for Image Classification 

1. **CNN**: This serves as a foundational model to establish image classification performance. 
2. **CNN-BN**: Designing a custom architecture to improve convergence and explores advanced techniques to demonstrate performance improvements.
3. **ResNet**: Implementing residual connections to enable training deeper networks by mitigating vanishing gradient problem in deeper networks.
4. **ResNeXt**: highlights the impact of grouped convolutions and cardinality on model performance, providing a modern alternative to traditional ResNet architectures.
5. **DenseNet**: demonstrates the effectiveness of feature reuse and compact architectures, It connects each layer to every other layer in a feed-forward fashion and utiliozes dense connections, reducing number of parameters.

- **Data Preparation**: Images and corresponding classifications are split into train and test sets.
- **Data Augmentation**: Different augmentation techniques such as rotation, translation, etc are performed to enhance the model's robustness.
- **Model Implementation**: Different models are trained using custom architectures to monitor training and testing accuracy(i.e `loss over epochs`) .
- **Evaluation**: Comparing results across all architectures. Generating visualizations(`loss over epochs`) for training/validation accuracy and loss.

## Installation

### Pb-lite Edge Detection

### Prerequisites

Ensure you have Python 3.8 or later installed. Install additional dependencies listed in the `requirements_pblite.txt` file.

### Setup Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/StarkGoku10/Pb_lite_Edgedetection.git
   cd pb_lite_edgedetection
   ```

2. **Create a Virtual Environment(Optional but recommended)**
    ```bash 
    python3 -m venv venv
    source venv/bin/activate #on windows: venv\Scripts\activate
    ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_pblite.txt
   ```

3. **Set up the folder structure for results**:
   ```bash
   python3 folders.py
   ```

4. **Dataset**:
    - This project is tested on BSDS500 Dataset, you can download and keep your dataset in the `Datasets/` directory.

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

