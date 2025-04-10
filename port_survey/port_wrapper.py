# PortMonitoring.py
import os
import torch
import cv2 as cv
from ultralytics import YOLO
from camera import Camera
import easyocr
# Set device to MPS on Mac if available, else CPU
device = torch.device("mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print("Model runs on", device.type)

class PortMonitoring(Camera):
    def __init__(self, boat_model, hin_model=None, encroachment_model=None):
        super().__init__()
        self.boat_model = YOLO(boat_model)
        self.boat_model.to(device)

        self.hin_model = YOLO(hin_model).to(device) if hin_model else None
        self.encroachment_model = YOLO(encroachment_model).to(device) if encroachment_model else None

        self.reader = easyocr.Reader(['en'])  # For OCR
    def extract_hin_number(self, image, hin_results):
        if not hin_results or len(hin_results[0].boxes) == 0:
            return None
        
        # Take the first detected HIN bounding box
        box = hin_results[0].boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        # Run OCR on the cropped region
        results = self.reader.readtext(roi)
        return results[0][1] if results else None
    
    def get_json_data(self):
        # Get the current frame from the camera
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        boat_results = self.boat_model(img)
        boat_count = len(boat_results[0])

        hin_detected = False
        hin_number = None
        if self.hin_model:
            hin_results = self.hin_model(img)
            hin_detected = len(hin_results[0]) > 0
            hin_number = self.extract_hin_number(img, hin_results)

        encroachment_detected = False
        if self.encroachment_model:
            encroachment_results = self.encroachment_model(img)
            encroachment_detected = len(encroachment_results[0]) > 0

        return {
            "boat_detected": boat_count > 0,
            "boat_count": boat_count,
            "hin_detected": hin_detected,
            "hin_number": hin_number,  # Added HIN number here
            "encroachment_detected": encroachment_detected
        }
    
    def process_image(self, image_path):
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        boat_results = self.boat_model(img)
        boat_count = len(boat_results[0])

        hin_detected = False
        hin_number = None
        if self.hin_model:
            hin_results = self.hin_model(img)
            hin_detected = len(hin_results[0]) > 0
            hin_number = self.extract_hin_number(img, hin_results)

        encroachment_detected = False
        if self.encroachment_model:
            encroachment_results = self.encroachment_model(img)
            encroachment_detected = len(encroachment_results[0]) > 0

        return {
            "boat_detected": boat_count > 0,
            "boat_count": boat_count,
            "hin_detected": hin_detected,
            "hin_number": hin_number,  # Added HIN number here
            "encroachment_detected": encroachment_detected
        }

# Example usage
if __name__ == "__main__":
    try:
        # Get the base directory path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Model paths (adjust according to your directory structure)
        boat_model_path = os.path.join(BASE_DIR, "classify", "best.pt")
        hin_model_path = os.path.join(BASE_DIR, "HIN", "best.pt")
        
        # Create port monitoring instance
        port_monitor = PortMonitoring(
            boat_model=boat_model_path,
            hin_model=hin_model_path
        )
        
        # Test with a sample image
        test_image = os.path.join(BASE_DIR, "HIN", "18.jpg")
        results = port_monitor.process_image(test_image)
        
        # Display results
        print("Results:", results)
        
    except Exception as e:
        print(f"Error during testing: {e}")