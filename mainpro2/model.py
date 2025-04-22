import numpy as np
import cv2

class CrowdDensityModel:
    """
    A crowd density estimation model based on image analysis using computer vision.
    The model estimates crowd density into Low, Medium, High, or Extremely High categories
    and provides person count estimates.
    """
    
    def __init__(self):
        """Initialize the crowd density estimation model."""
        self.class_names = ["Low", "Medium", "High", "Extremely High"]
        
    def predict(self, preprocessed_image):
        """
        Predict crowd density from a preprocessed image.
        
        Args:
            preprocessed_image: Preprocessed input image
            
        Returns:
            density_class: One of "Low", "Medium", "High", or "Extremely High"
            confidence_scores: List of confidence values for each class
            person_count: Estimated number of people in the image
            density_map: Heatmap representing crowd density
        """
        # Extract the actual image from the batch dimension
        image = preprocessed_image[0]
        
        # Calculate edge density
        edge_density = np.mean(image[:, :, 0])
        
        # Calculate spatial density (percentage of non-zero pixels)
        spatial_density = np.count_nonzero(image[:, :, 0]) / image[:, :, 0].size
        
        # Estimate person count based on edge density and spatial density
        complexity_score = edge_density + spatial_density
        person_count = int(complexity_score * 100)
        
        # Create a density map based on the edge density
        density_map = self._generate_density_map(image)
        
        # Determine class based on person count
        if person_count < 15:
            class_idx = 0  # Low density
            confidence = [0.8, 0.1, 0.05, 0.05]
        elif person_count < 50:
            class_idx = 1  # Medium density
            confidence = [0.1, 0.8, 0.05, 0.05]
        elif person_count < 100:
            class_idx = 2  # High density
            confidence = [0.05, 0.1, 0.8, 0.05]
        else:
            class_idx = 3  # Extremely High density
            confidence = [0.05, 0.05, 0.1, 0.8]
        
        # Normalize confidence scores
        confidence = np.array(confidence)
        confidence = confidence / confidence.sum()
        
        # Get the class label
        density_class = self.class_names[class_idx]
        
        return density_class, confidence.tolist(), person_count, density_map
    
    def _generate_density_map(self, image):
        """
        Generate a density map based on image features.
        
        Args:
            image: Input processed image
            
        Returns:
            density_map: A heatmap representing crowd density (0-1 range)
        """
        single_channel = image[:, :, 0]
        
        # Apply Gaussian blur to smooth the map
        blurred = cv2.GaussianBlur(single_channel, (15, 15), 0)
        
        # Normalize to 0-1 range
        if blurred.max() > 0:
            density_map = blurred / blurred.max()
        else:
            density_map = blurred
        
        return density_map