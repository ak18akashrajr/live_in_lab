# ForestDetection.py
import os
import cv2
import torch
from ultralytics import YOLO
from camera import Camera  # Assumes you have a Camera base class that supplies frames

# Get the base directory path (location of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set device to MPS on Mac (if available), CUDA if available, or CPU by default
device = torch.device("mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print("Model runs on", device.type)


class DeforestationDetection(Camera):
    """
    This class handles deforestation detection using a YOLO model.
    """
    def __init__(self, model_path=None):
        super().__init__()
        # Use provided model path or default to relative path based on directory structure
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "forest", "deforestation", "yolov8n.pt")
        self.model = YOLO(model_path).to(device)
    
    def get_deforestation_data(self):
        """
        Captures a frame from the camera and runs the YOLO model for deforestation detection.
        Returns the annotated image.
        """
        # Get the current frame from the camera
        img = self.get_frame()
        # Run the YOLO model on the image
        result = self.model(img, show_conf=False)
        # Plot and get the annotated result image
        result_img = result[0].plot()
        return result_img
    
    def process_image(self, image_path):
        """
        Process a single image file instead of using the camera.
        Useful for testing with existing images.
        """
        # Read the image file
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or could not be read: {image_path}")
        
        # Run the YOLO model on the image
        result = self.model(img, show_conf=False)
        # Plot and get the annotated result image
        result_img = result[0].plot()
        
        # Save the result in the same directory as the input
        output_path = os.path.join(os.path.dirname(image_path), "result_img.jpg")
        cv2.imwrite(output_path, result_img)
        
        return result_img


class WildfireDetection(Camera):
    """
    This class handles wildfire detection using a YOLO model.
    """
    def __init__(self, model_path=None):
        super().__init__()
        # Use provided model path or default to relative path based on directory structure
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "forest", "wildfire", "yolov8n.pt")
        self.model = YOLO(model_path).to(device)
    
    def get_wildfire_data(self):
        """
        Captures a frame from the camera and runs the YOLO model for wildfire detection.
        Returns the annotated image.
        """
        # Get the current frame from the camera
        img = self.get_frame()
        # Run the YOLO model on the frame
        result = self.model(img, show_conf=False)
        # Get the result image with predictions drawn
        result_img = result[0].plot()
        return result_img
    
    def process_image(self, image_path):
        """
        Process a single image file instead of using the camera.
        Useful for testing with existing images.
        """
        # Read the image file
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or could not be read: {image_path}")
        
        # Run the YOLO model on the image
        result = self.model(img, show_conf=False)
        # Plot and get the annotated result image
        result_img = result[0].plot()
        
        # Save the result in the same directory as the input
        output_path = os.path.join(os.path.dirname(image_path), "result_img.jpg")
        cv2.imwrite(output_path, result_img)
        
        return result_img


# Example usage (for testing purposes):
if __name__ == "__main__":
    try:
        # Use relative paths based on the directory structure
        deforestation_model_path = os.path.join(BASE_DIR, "forest", "deforestation", "yolov8n.pt")
        wildfire_model_path = os.path.join(BASE_DIR, "forest", "wildfire", "yolov8n.pt")
        
        # For testing with sample images instead of camera
        deforestation_test_image = os.path.join(BASE_DIR, "forest", "deforestation", "input.jpg")
        wildfire_test_image = os.path.join(BASE_DIR, "forest", "wildfire", "input.jpg")
        
        # Create detector instances
        deforestation_detector = DeforestationDetection(model_path=deforestation_model_path)
        wildfire_detector = WildfireDetection(model_path=wildfire_model_path)
        
        # Test with sample images
        print("Processing deforestation test image...")
        deforestation_result = deforestation_detector.process_image(deforestation_test_image)
        
        print("Processing wildfire test image...")
        wildfire_result = wildfire_detector.process_image(wildfire_test_image)
        
        # Display the results
        cv2.imshow("Deforestation Detection", deforestation_result)
        cv2.imshow("Wildfire Detection", wildfire_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Testing complete!")
        
    except Exception as e:
        print(f"Error during testing: {e}")