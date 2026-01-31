"""
Streamlit App for Colorectal Cell Classification
A user-friendly interface for testing the deep learning model
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import urllib.request

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.image import resize_with_pad
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Colorectal Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Note: Cell types are determined dynamically from the dataset to match model order
# pd.get_dummies creates columns in alphabetical order, so we'll get the actual order from data

@st.cache_data
def download_data_if_needed():
    """Download data files if they don't already exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    
    download_url_prefix = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/"
    
    if images_path.exists() and labels_path.exists():
        return str(images_path), str(labels_path)
    
    if not images_path.exists() or not labels_path.exists():
        with st.spinner("Downloading data files (~500MB). This may take a few minutes..."):
            images_url = download_url_prefix + "images.npy"
            labels_url = download_url_prefix + "labels.npy"
            
            if not images_path.exists():
                urllib.request.urlretrieve(images_url, images_path)
            if not labels_path.exists():
                urllib.request.urlretrieve(labels_url, labels_path)
    
    return str(images_path), str(labels_path)

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size for model input."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in correct format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    
    # Convert to tensor and resize using TensorFlow's resize_with_pad
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # Use TensorFlow's resize_with_pad for consistency with training
    image_tf = tf.constant(image)
    resized_tf = resize_with_pad(image_tf, target_size[0], target_size[1], antialias=True)
    resized = resized_tf.numpy().astype(np.uint8)
    
    return resized

def build_model():
    """Build the EfficientNetB0 model architecture."""
    # Load EfficientNetB0 base model
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    
    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    output = Dense(8, activation="softmax")(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "categorical_crossentropy"]
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=5):
    """Train the model and return history."""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    return history

def ResizeImages(images, height, width):
    """Resize images to target dimensions."""
    return np.array([resize_with_pad(image, height, width, antialias=True) for image in images]).astype(int)

@st.cache_data
def load_training_data():
    """Load training data for test samples."""
    try:
        images_path, labels_path = download_data_if_needed()
        images = np.load(images_path)
        labels = np.load(labels_path)
        
        # Get unique labels sorted alphabetically (same order as pd.get_dummies creates)
        unique_labels = sorted(list(set(labels)))
        
        # Create label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        return images, labels, unique_labels, label_to_idx
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def predict_image(model, image_array):
    """Make prediction on an image."""
    # Resize image
    resized_img = resize_image(image_array)
    
    # Match training format: ResizeImages returns astype(int) [0, 255]
    # Training passes X_train_resized directly which is astype(int) [0, 255]
    # Convert to float32 but keep [0, 255] range to match training
    img_for_prediction = resized_img.astype(np.float32)
    
    # Expand dimensions for batch
    img_batch = np.expand_dims(img_for_prediction, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    
    return predictions, resized_img

def display_prediction_results(predictions, cell_types):
    """Display prediction results in a user-friendly format."""
    # Debug: print raw predictions
    # st.write("Debug - Raw predictions:", predictions)
    # st.write("Debug - CELL_TYPES:", cell_types)
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    top_class = cell_types[top_idx] if top_idx < len(cell_types) else f"Unknown({top_idx})"
    top_confidence = predictions[top_idx]
    
    # Create DataFrame for all predictions
    df = pd.DataFrame({
        'Cell Type': cell_types,
        'Confidence': predictions
    }).sort_values('Confidence', ascending=False)
    
    # Display top prediction prominently
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 Predicted Cell Type: <strong>{top_class}</strong></h2>
        <h3>Confidence: <strong>{top_confidence*100:.2f}%</strong></h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all predictions
    st.subheader("📊 All Predictions")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#4CAF50' if i == top_idx else '#2196F3' for i in range(len(cell_types))]
        ax.barh(df['Cell Type'], df['Confidence'] * 100, color=colors)
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Prediction Confidence Scores')
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Data table
        df_display = df.copy()
        df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x*100:.2f}%")
        df_display = df_display.reset_index(drop=True)
        st.dataframe(df_display, use_container_width=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Colorectal Cell Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Model for Histology Cell Type Classification</p>', unsafe_allow_html=True)
    
    # Check for saved model
    model_path = Path("models/trained_model.weights.h5")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check model status
    model_trained = model_path.exists()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading option
        st.subheader("Model Options")
        
        if model_trained:
            st.success("✅ Trained model found!")
        else:
            st.warning("⚠️ Model not trained yet")
        
        train_new_model = st.checkbox("Train New Model (takes ~5-10 minutes)", value=False)
        
        if model_trained:
            load_pretrained = st.checkbox("Use Pre-trained Model", value=True)
        else:
            load_pretrained = False
        
        st.divider()
        
        st.subheader("📖 About")
        st.markdown("""
        This application uses **EfficientNetB0** with transfer learning 
        to classify colorectal histology images into 8 cell types:
        
        - Adipose
        - Complex  
        - Debris
        - Empty
        - Lympho
        - Mucosa
        - Stroma
        - Tumor
        """)
        
        st.divider()
        
        st.markdown("### 🔗 Links")
        st.markdown("[GitHub Repository](https://github.com)")
        st.markdown("[Documentation](https://github.com)")
    
    # Load or build model
    model = build_model()
    
    # Load pre-trained weights if available
    if load_pretrained and model_path.exists():
        try:
            with st.spinner("Loading trained model weights..."):
                model.load_weights(str(model_path))
            st.success("✅ Trained model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Using untrained model. Predictions will not be accurate.")
    elif not model_path.exists():
        st.info("ℹ️ No trained model found. Model is untrained - predictions will not be accurate.")
    
    # Load training data for test samples
    images, labels, unique_labels, label_to_idx = load_training_data()
    
    # Get cell types in the correct order (alphabetical, matching pd.get_dummies order)
    # This ensures predictions map to correct class names
    CELL_TYPES = unique_labels if unique_labels is not None else ['Adipose', 'Complex', 'Debris', 'Empty', 'Lympho', 'Mucosa', 'Stroma', 'Tumor']
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🖼️ Predict", "📸 Test Samples", "📈 Model Info"])
    
    with tab1:
        st.header("Image Classification")
        st.markdown("Upload an image or use a test sample to classify colorectal histology cell types.")
        
        # Image input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Test Sample"],
            horizontal=True
        )
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a histology image for classification"
            )
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                
                # Display original image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📤 Uploaded Image")
                    st.image(image, caption="Original Image", use_container_width=True)
                
                with col2:
                    # Make prediction
                    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            predictions, processed_img = predict_image(model, np.array(image))
                            display_prediction_results(predictions, CELL_TYPES)
                            
                            # Show processed image
                            st.subheader("🔄 Processed Image")
                            st.image(processed_img, caption="Preprocessed for Model (224x224)", use_container_width=True)
        
        else:  # Use Test Sample
            if images is not None and labels is not None:
                st.subheader("Select a Test Sample")
                
                # Create sample selector
                sample_indices = list(range(len(images)))
                sample_idx = st.selectbox(
                    "Choose a test sample:",
                    options=sample_indices,
                    format_func=lambda x: f"Sample {x} - {labels[x]}"
                )
                
                # Display selected sample
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📸 Selected Sample")
                    selected_image = images[sample_idx]
                    actual_label = labels[sample_idx]
                    st.image(selected_image, caption=f"Actual Label: {actual_label}", use_container_width=True)
                
                with col2:
                    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            predictions, processed_img = predict_image(model, selected_image)
                            
                            # Show actual vs predicted
                            top_idx = np.argmax(predictions)
                            predicted_label = CELL_TYPES[top_idx]
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Actual: <strong>{actual_label}</strong></h3>
                                <h3>Predicted: <strong>{predicted_label}</strong></h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            display_prediction_results(predictions, CELL_TYPES)
                            
                            # Show processed image
                            st.subheader("🔄 Processed Image")
                            st.image(processed_img, caption="Preprocessed for Model (224x224)", use_container_width=True)
            else:
                st.warning("Test samples not available. Please ensure data files are downloaded.")
    
    with tab2:
        st.header("Test Sample Gallery")
        
        if images is not None and labels is not None:
            # Group by cell type
            cell_type_counts = pd.Series(labels).value_counts().sort_index()
            
            st.subheader("Dataset Statistics")
            st.dataframe(cell_type_counts.reset_index().rename(columns={'index': 'Cell Type', 0: 'Count'}))
            
            # Show samples by cell type
            selected_type = st.selectbox("Filter by Cell Type:", ["All"] + sorted(list(set(labels))))
            
            # Filter images
            if selected_type == "All":
                display_images = images
                display_labels = labels
            else:
                mask = np.array([label == selected_type for label in labels])
                display_images = images[mask]
                display_labels = np.array(labels)[mask]
            
            # Display grid of images
            cols_per_row = 4
            num_images = len(display_images)
            
            for i in range(0, min(num_images, 12), cols_per_row):  # Show max 12 images
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < num_images:
                        with col:
                            st.image(display_images[i + j], caption=display_labels[i + j], use_container_width=True)
        else:
            st.warning("Test samples not available. Please ensure data files are downloaded.")
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏗️ Architecture")
            st.markdown("""
            **Base Model**: EfficientNetB0
            - Pre-trained on ImageNet
            - Frozen base layers
            - Input: 224×224×3 images
            
            **Custom Layers**:
            - Flatten layer
            - Dense layer (8 classes, softmax)
            """)
            
            st.subheader("📊 Performance")
            st.markdown("""
            - **Training Accuracy**: ~99%
            - **Validation Accuracy**: ~86%
            - **Training Time**: ~5 minutes (5 epochs)
            - **Transfer Learning**: Yes (EfficientNetB0)
            """)
        
        with col2:
            st.subheader("🔧 Technical Details")
            st.markdown("""
            - **Framework**: TensorFlow/Keras
            - **Optimizer**: Adam
            - **Loss Function**: Categorical Crossentropy
            - **Train/Test Split**: 80/20
            - **Image Preprocessing**: Resize to 224×224, normalize
            """)
            
            st.subheader("📦 Dataset")
            st.markdown("""
            - **Source**: TensorFlow Datasets (colorectal_histology)
            - **Cell Types**: 8 classes
            - **Image Size**: Variable (resized to 224×224)
            """)
        
        # Model summary
        if st.checkbox("Show Model Summary"):
            st.code(model.summary(), language=None)

if __name__ == "__main__":
    main()
