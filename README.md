# Sign Language Recognition with PyTorch CNN

This project is a simple, practical introduction to image classification with PyTorch.
The goal is to teach a model to recognize American Sign Language (ASL) hand signs and map them to alphabet letters.

Instead of being a large production system, this repository is built as a learning project: load images, preprocess them, train a CNN, and check how well it predicts.

## The Story of This Project

When you open this project, you are walking through a complete deep-learning workflow:

1. **Start with labeled hand-sign images** (one folder per letter).
2. **Clean and standardize the images** so the model gets consistent input.
3. **Train a CNN** to learn visual patterns from each sign.
4. **Evaluate on test images** to see how accurately the model generalizes.

That’s it—clear, focused, and easy to extend.

## What You’ll Find Here

- `ASL.ipynb` — the main notebook for training and evaluating the CNN.
- `crop.ipynb` — notebook for image preparation/cropping workflow.
- `README.md` — this guide.

## Dataset Layout

The notebooks expect a folder structure similar to this:

```text
data/
└── asl_alphabet/
    ├── train/
    │   ├── A/
    │   ├── B/
    │   └── ...
    └── test/
        ├── A/
        ├── B/
        └── ...
```

Each letter folder contains images for that class.

## How the Model Works (Simple View)

The `SignLanguageCNN` model follows a common pattern:

- **Convolution + ReLU + Pooling** layers extract visual features from hand images.
- A **Flatten** step converts feature maps into a vector.
- **Fully connected layers** produce final class scores for each letter.

Training uses:

- `CrossEntropyLoss` for multi-class classification.
- `Adam` optimizer for gradient updates.

## Requirements

Install the main dependencies:

```bash
pip install torch torchvision
```

(`Pillow` is usually installed automatically with `torchvision`.)

## Quick Start

1. Place the ASL dataset in `data/asl_alphabet/` using the structure above.
2. Open `ASL.ipynb`.
3. Run cells from top to bottom:
   - load and transform data,
   - build model,
   - train for a few epochs,
   - evaluate test accuracy.

## Expected Output

After training, the notebook prints metrics such as loss and accuracy.
Exact numbers will vary by dataset version, random seed, and hardware.

## Why This Repository Is Useful

If you are learning computer vision with PyTorch, this project gives you an end-to-end reference that is small enough to understand quickly and flexible enough to improve with your own ideas.
