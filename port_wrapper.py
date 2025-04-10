# PortMonitoring.py
import os
import torch
import cv2 as cv
from ultralytics import YOLO
from camera import Camera

# Set device to MPS on Mac if available, else CPU
device = torch.device("mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print("Model runs on", device.type)

class PortMonitoring(Camera):
    def __init__(self, boat_model, hin_model=None, encroachment_model=None):
        super().__init__()
        self.boat_model = YOLO(boat_model).to(device)
        
        if hin_model:
            self.hin_model = YOLO(hin_model).to(device)
        else:
            self.hin_model = None
            
        if encroachment_model:
            self.encroachment_model = YOLO(encroachment_model).to(device)
        else:
            self.encroachment_model = None
    
    def get_json_data(self):
        # Get the current frame from the camera
        img = self.get_frame()
        
        # Run boat detection
        boat_results = self.boat_model(img)
        boat_count = len(boat_results[0])
        
        # Run HIN detection if model is available
        hin_detected = False
        if self.hin_model:
            hin_results = self.hin_model(img)
            hin_detected = len(hin_results[0]) > 0
        
        # Run encroachment detection if model is available
        encroachment_detected = False
        if self.encroachment_model:
            encroachment_results = self.encroachment_model(img)
            encroachment_detected = len(encroachment_results[0]) > 0
            
        return {
            "boat_detected": boat_count > 0,
            "boat_count": boat_count,
            "hin_detected": hin_detected,
            "encroachment_detected": encroachment_detected
        }
    
    def process_image(self, image_path):
        # Read the image file
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")
        
        # Run boat detection
        boat_results = self.boat_model(img)
        boat_count = len(boat_results[0])
        
        # Run HIN detection if model is available
        hin_detected = False
        if self.hin_model:
            hin_results = self.hin_model(img)
            hin_detected = len(hin_results[0]) > 0
        
        # Run encroachment detection if model is available
        encroachment_detected = False
        if self.encroachment_model:
            encroachment_results = self.encroachment_model(img)
            encroachment_detected = len(encroachment_results[0]) > 0
            
        return {
            "boat_detected": boat_count > 0,
            "boat_count": boat_count,
            "hin_detected": hin_detected,
            "encroachment_detected": encroachment_detected
        }

# Example usage
if __name__ == "__main__":
    try:
        # Get the base directory path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Model paths (adjust according to your directory structure)
        boat_model_path = os.path.join(BASE_DIR, "classify_boats", "best.pt")
        hin_model_path = os.path.join(BASE_DIR, "HIN", "best.pt")
        
        # Create port monitoring instance
        port_monitor = PortMonitoring(
            boat_model=boat_model_path,
            hin_model=hin_model_path
        )
        
        # Test with a sample image
        test_image = os.path.join(BASE_DIR, "test_images", "input.jpg")
        results = port_monitor.process_image(test_image)
        
        # Display results
        print("Results:", results)
        
    except Exception as e:
        print(f"Error during testing: {e}")