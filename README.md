# Sign Language Recognition with PyTorch CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to recognize signs from the ASL Alphabet Dataset. The goal is to classify images of hand gestures into their corresponding English alphabet letters.

## Features
-   **Data Loading**: Utilizes `torchvision.datasets.ImageFolder` for efficient loading of image data organized into class-specific folders.
-   **Image Preprocessing**: Applies transformations such as resizing, grayscale conversion,ToTensor, and normalization to prepare images for the CNN.
-   **CNN Model**: A custom-designed CNN architecture in PyTorch for image classification.
-   **Training Loop**: Implements a standard training loop with `CrossEntropyLoss` and `Adam` optimizer.
-   **Evaluation**: Measures the model's accuracy on a separate test set.

## Dataset
This project is designed to work with a dataset structured like the ASL Alphabet Dataset (e.g., from Kaggle). The expected directory structure is:

```
data/
├── asl_alphabet/
│   ├── train/
│   │   ├── A/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   ├── B/
│   │   └── ...
│   └── test/
│       ├── A/
│       │   ├── image_test1.jpg
│       │   └── ...
│       ├── B/
│       └── ...
```

Each subfolder (e.g., `A`, `B`) within `train` and `test` represents a distinct class (a letter of the ASL alphabet).

## Requirements
To run this project, you will need the following Python libraries:
-   `torch`
-   `torchvision`
-   `Pillow` (often installed with `torchvision`)

These can typically be installed via pip:
```bash
pip install torch torchvision
```

## Model Architecture
The `SignLanguageCNN` model consists of:
1.  Two convolutional layers with ReLU activation and Max Pooling.
    -   First layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel.
    -   Second layer: 32 input channels, 64 output channels, 3x3 kernel.
2.  A `Flatten` layer to convert the 2D feature maps into a 1D vector.
3.  Two fully connected (linear) layers with ReLU activation for the hidden layer.
    -   First linear layer: maps from `64 * 16 * 16` (after pooling) to 256 features.
    -   Second linear layer: maps from 256 features to `num_classes` (26 for ASL alphabet).

## Usage
1.  **Prepare the Dataset**: Ensure your ASL Alphabet dataset is organized into `train` and `test` directories within `./data/asl_alphabet/`, with subfolders for each letter as described above.
2.  **Run the Notebook**: Execute the code cells sequentially. The script will:
    -   Load and preprocess the dataset.
    -   Initialize the CNN model, loss function, and optimizer.
    -   Train the model for 5 epochs.
    -   Evaluate the model on the test set and print the accuracy.
```
(The specific loss and accuracy values will vary based on hardware, randomness, and dataset characteristics.)
