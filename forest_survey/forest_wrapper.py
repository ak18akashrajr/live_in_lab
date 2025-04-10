# ForestDetection.py
import os
import torch
import cv2
from ultralytics import YOLO
from camera import Camera

# Set device to MPS on Mac if available, else CPU
device = torch.device("mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print("Model runs on", device.type)

class ForestMonitoring(Camera):
    """
    This class handles forest monitoring including deforestation and wildfire detection.
    """
    def __init__(self, deforestation_model, wildfire_model):
        super().__init__()
        # Load the models
        self.deforestation_model = YOLO(deforestation_model).to(device)
        self.wildfire_model = YOLO(wildfire_model).to(device)
    
    def get_json_data(self):
        """
        Captures a frame from the camera and runs both detection models.
        Returns a JSON with detection results.
        """
        # Get the current frame from the camera
        img = self.get_frame()
        
        # Run the deforestation detection model
        deforestation_results = self.deforestation_model(img)
        
        # Run the wildfire detection model
        wildfire_results = self.wildfire_model(img)
        
        # Check if deforestation is detected (if any objects are detected)
        deforestation_detected = len(deforestation_results[0]) > 0
        
        # Check if wildfire is detected (if any objects are detected)
        wildfire_detected = len(wildfire_results[0]) > 0
        
        return {
            "deforestation_detected": deforestation_detected,
            "wildfire_detected": wildfire_detected,
            "deforestation_count": len(deforestation_results[0]),
            "wildfire_count": len(wildfire_results[0])
        }
    
    def process_image(self, image_path):
        """
        Process a single image file instead of using the camera.
        Returns a JSON with detection results.
        """
        # Read the image file
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or could not be read: {image_path}")
        
        # Run the deforestation detection model
        deforestation_results = self.deforestation_model(img)
        
        # Run the wildfire detection model
        wildfire_results = self.wildfire_model(img)
        
        # Check if deforestation is detected
        deforestation_detected = len(deforestation_results[0]) > 0
        
        # Check if wildfire is detected
        wildfire_detected = len(wildfire_results[0]) > 0
        
        return {
            "deforestation_detected": deforestation_detected,
            "wildfire_detected": wildfire_detected,
            "deforestation_count": len(deforestation_results[0]),
            "wildfire_count": len(wildfire_results[0])
        }

# Example usage
if __name__ == "__main__":
    try:
        # Get the base directory path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Set model paths
        deforestation_model_path = os.path.join(BASE_DIR, "forest", "deforestation", "yolov8n.pt")
        wildfire_model_path = os.path.join(BASE_DIR, "forest", "wildfire", "yolov8n.pt")
        
        # Create forest monitoring instance
        forest_monitor = ForestMonitoring(
            deforestation_model=deforestation_model_path,
            wildfire_model=wildfire_model_path
        )
        
        # For testing with a sample image
        test_image = os.path.join(BASE_DIR, "forest", "deforestation", "input.jpg")
        
        # Get detection results
        print("Processing test image...")
        results = forest_monitor.process_image(test_image)
        
        # Display results
        print("Results:", results)
        
    except Exception as e:
        print(f"Error during testing: {e}")