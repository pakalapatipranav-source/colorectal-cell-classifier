# Colorectal Cell Classifier
Colorectal cancer is one of the leading causes of cancer-related deaths worldwide, and histology images are a key tool for understanding tumor structure at the cellular level. In this project, I developed a machine learning pipeline to classify different cell types in colorectal cancer tissue using histopathological image data.

The goal of this project was to explore how computer vision and deep learning techniques can be applied to real biomedical datasets, and to understand the practical challenges involved in medical image classification rather than to build a clinically deployable system.

## Overview
This project implements a convolutional neural network for automated classification of histology cell types, which is crucial for cancer research and diagnosis. By leveraging transfer learning with EfficientNetB0, the model achieves high accuracy while requiring minimal training time and computational resources.

## Why This Matters

Cell-type classification in histology images is a critical task in computational pathology. Automating this process enables:
- **Large-scale quantitative analysis** of tissue composition
- **Consistent and reproducible** cell type identification
- **Support for cancer research** and diagnostic workflows
- **Time-efficient analysis** of large histology datasets

I chose this problem because it combines machine learning, image processing, and biomedical science, allowing me to apply core CS concepts to a real-world, high-impact domain.

## Features

- **Transfer Learning Implementation**: Leverages EfficientNetB0 pre-trained on ImageNet for robust feature extraction
- **8-Class Classification**: Identifies multiple colorectal histology cell types:
  - Adipose, Complex, Debris, Empty, Lympho, Mucosa, Stroma, Tumor
- **Comprehensive Evaluation**: Includes accuracy tracking, confusion matrices, and visual predictions
- **Multiple Interfaces**: Available as Python script, Jupyter notebook, and Streamlit web app
- **Interactive Web Interface**: User-friendly Streamlit app for easy testing and demonstration
- **Automated Data Management**: Downloads and caches data automatically on first run
- **Production-Ready**: Clean, well-documented code suitable for local execution and deployment

## Implementation Details

### Architecture

This project uses **transfer learning** with EfficientNetB0 as the base model. The pre-trained EfficientNetB0 model (trained on ImageNet) is frozen to preserve learned features, and custom dense layers are added on top for the 8-class classification task.

**Model Architecture:**
- **Base Model**: EfficientNetB0 (frozen, pre-trained on ImageNet)
- **Custom Layers**: Flatten → Dense layer with softmax activation (8 classes)
- **Input Size**: 224×224 pixels (resized from original histology images)
- **Optimizer**: Adam
- **Loss Function**: Categorical crossentropy
- **Training**: 5 epochs with 80/20 train-test split

### Workflow

1. **Data Loading**: Downloads histology images and labels from remote source
2. **Preprocessing**: Resizes images to 224×224, applies one-hot encoding to labels
3. **Model Building**: Loads pre-trained EfficientNetB0, adds custom classification head
4. **Training**: Trains on 80% of data, validates on 20%
5. **Evaluation**: Generates confusion matrix and visualization plots
6. **Prediction**: Displays sample predictions with confidence scores

## Project Structure

```
colorectal-cell-classifier/
├── app.py  # Streamlit web application
├── train_model.py  # Model training script
├── notebooks/
│   └── colorectal_cell_classification.ipynb  # Jupyter notebook with the full pipeline
├── src/
│   └── classifying_the_different_cells_in_colorectal_cancer.py  # Python script version
├── requirements.txt  # Python dependencies
├── .gitignore  # Git ignore rules
└── README.md  # This file
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project requires:
- TensorFlow (for deep learning models)
- TensorFlow Datasets (for data loading)
- NumPy, Pandas (for data manipulation)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn, Scikit-image (for utilities and preprocessing)
- Pillow, Requests (for image handling)

## Running Locally

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd colorectal-cell-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3.12 -m venv venv  # or python3.11
   source venv/bin/activate  # On macOS/Linux
   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This project requires Python 3.11 or 3.12 (TensorFlow doesn't support Python 3.14+ yet). If your default `python3` is 3.14+, use `python3.12` or `python3.11` instead.

### Running the Code

The project includes both a Python script and a Jupyter notebook. You can use either:

**Option 1: Python Script**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
python src/classifying_the_different_cells_in_colorectal_cancer.py
```

**Option 2: Jupyter Notebook**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
jupyter notebook notebooks/colorectal_cell_classification.ipynb
```

**Option 3: Streamlit Web App (Interactive UI)**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Run the Streamlit app
streamlit run app.py
```

The Streamlit app will open in your default web browser at `http://localhost:8501`. This provides an interactive interface for:
- Uploading images for classification
- Testing with sample images from the dataset
- Viewing model predictions with confidence scores
- Exploring test sample gallery

**Note:** The Streamlit app will use a pre-trained model if available. To train the model first, see the "Training the Model" section below.

### Training the Model

Before using the Streamlit app with accurate predictions, you should train the model:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Run the training script
python train_model.py
```

This will:
- Download the dataset (if not already present)
- Prepare and preprocess the data
- Build the EfficientNetB0 model
- Train for 5 epochs (~5-10 minutes)
- Save the trained model weights to `models/trained_model.weights.h5`
- Generate a training history plot at `models/training_history.png`

The trained model will automatically be loaded by the Streamlit app on subsequent runs.

### Data Download

On first run, the script will automatically download the required data files (~500MB):
- Images: `data/images.npy`
- Labels: `data/labels.npy`

The data will be saved in a `data/` directory and reused on subsequent runs, so you won't need to download it again. The download may take a few minutes depending on your internet connection.

**Note:** The `data/` directory is excluded from git via `.gitignore` since the files are large. Each developer will download the data on their first run.

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **EfficientNetB0** - Pre-trained convolutional neural network for transfer learning
- **Streamlit** - Web application framework for interactive UI
- **NumPy/Pandas** - Data manipulation and preprocessing
- **Matplotlib/Seaborn** - Data visualization and result plotting
- **Scikit-learn** - Machine learning utilities
- **Jupyter Notebooks** - Interactive development and experimentation

## Results

The model demonstrates strong classification performance on the colorectal histology dataset. After 5 epochs of training:

- **Training Accuracy**: ~99% (final epoch)
- **Validation Accuracy**: ~86% (final epoch)
- **Model Architecture**: EfficientNetB0 base with custom dense layer
- **Training Time**: Fast convergence thanks to transfer learning

The transfer learning approach with EfficientNetB0 allows for:
- Efficient training with minimal epochs
- Strong generalization on limited medical imaging data
- Leveraging pre-trained ImageNet features for histology classification

### Evaluation Metrics

The project includes comprehensive evaluation:
- **Training curves**: Track accuracy and loss over epochs
- **Confusion matrix**: Detailed per-class performance analysis
- **Sample predictions**: Visual display with confidence scores

## Key Highlights

- ✅ **Transfer Learning**: EfficientNetB0 adaptation for medical imaging
- ✅ **Multi-class Classification**: 8 distinct cell type categories
- ✅ **End-to-End Pipeline**: From data loading to visualization
- ✅ **Interactive Web App**: Streamlit interface for easy testing and demonstration
- ✅ **Well-Documented**: Clear code structure and comprehensive README
- ✅ **Reproducible**: Easy setup and execution instructions
- ✅ **Production-Ready**: Supports script, notebook, and web app workflows

## What I Would Improve With More Time

If I were to continue this project, I would:
- Apply data augmentation to increase dataset diversity  
- Use transfer learning with pre-trained convolutional models  
- Perform more extensive hyperparameter tuning  
- Explore model interpretability techniques (e.g., saliency or activation maps)  
- Build a lightweight inference demo to allow users to upload and classify images interactively  

These steps would help transition the project from an exploratory analysis to a more robust research-oriented system.

## License

This project is open source and available for educational purposes.
