import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for the crowd density estimation model.
    
    Args:
        image (numpy.ndarray): Input image in RGB format
        target_size (tuple): Target size for model input (default: (224, 224))
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Resize to target size
    resized = cv2.resize(edges, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    
    # Convert back to 3 channels (required for model input)
    preprocessed = np.stack([normalized] * 3, axis=-1)
    
    # Add batch dimension
    batched = np.expand_dims(preprocessed, axis=0)
    
    return batched

def get_density_color(density_class):
    """
    Get the background color for a density class.
    
    Args:
        density_class (str): One of "Low", "Medium", or "High"
        
    Returns:
        str: Hex color code for the density class
    """
    if density_class == "Low":
        return "#28a745"  # Green
    elif density_class == "Medium":
        return "#fd7e14"  # Orange
    elif density_class == "High":
        return "#dc3545"  # Red
    else:
        return "#6c757d"  # Gray (default)