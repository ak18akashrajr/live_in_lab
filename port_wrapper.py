# PortDetection.py
import cv2
import numpy as np
import torch
import imageio
import os
from ultralytics import YOLO

# Get the base directory path (location of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# For HIN detection
from HIN.identify import find_hin
from HIN.decoder import decode

# For land encroachment detection
from land_encroachment.boat_id import is_boat
from land_encroachment.edge_detect import remove_boat_edge
from land_encroachment.compare import is_encroach

# Set up device (MPS on Mac if available, else CPU)
device = torch.device("mps" if hasattr(torch, 'mps') and torch.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print("Running on device:", device.type)


class BoatClassifier:
    """
    Classify boats using a YOLO model.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Use a relative path based on your directory structure
            model_path = os.path.join(BASE_DIR, "classify_boats", "runs", "detect", "train6", "weights", "best.pt")
        self.model = YOLO(model_path).to(device)
    
    def classify_boats(self, image_path):
        """
        Reads an image from the provided path, runs classification, and returns the result image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found at: " + image_path)
        
        result = self.model(img, show_conf=False)
        result_img = result[0].plot()  # Annotated image
        
        # Display the result (optional)
        cv2.imshow("Boat Classification", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the output to the current directory
        output_path = os.path.join(os.path.dirname(image_path), "boat_classification_result.jpg")
        cv2.imwrite(output_path, result_img)
        return result_img


class HINDetector:
    """
    Detect the HIN number on vessels using a YOLO model plus additional processing.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Use a relative path to the HIN model based on directory structure
            model_path = os.path.join(BASE_DIR, "HIN", "HIN_detection", "runs", "detect", "train4", "weights", "best.pt")
            # Fallback to best.pt if the specific path doesn't exist
            if not os.path.exists(model_path):
                model_path = os.path.join(BASE_DIR, "HIN", "best.pt")
        self.model = YOLO(model_path).to(device)
    
    def detect_hin(self, image_path):
        """
        Reads an image from the provided path, performs HIN detection, and returns the HIN value.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found at: " + image_path)
        
        # Use the YOLO model to predict and then draw black rectangles over detected areas.
        results = self.model.predict(image_path)
        for r in results:
            for box in r.boxes.xywh:
                x, y, width, height = box.tolist()
                x_min = int(x - width / 2)
                y_min = int(y - height / 2)
                x_max = int(x + width / 2)
                y_max = int(y + height / 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
        
        # Use additional detection logic (e.g., find_hin and decode)
        success, cropped_image = find_hin(img)
        hin_value = None
        if success:
            pred, result = decode(cropped_image)
            if pred:
                hin_value = result
        
        # Optionally display and/or save the processed image
        cv2.imshow("HIN Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return hin_value


def check_land_encroachment(image_path):
    """
    Processes the provided image to detect land encroachment.
    Returns True if encroachment is found, False otherwise.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at: " + image_path)
    
    boat_covered, there_boat = is_boat(image)
    output_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(output_dir, "boat_covered.jpg"), boat_covered)
    
    boat_removed = remove_boat_edge(boat_covered)
    cv2.imwrite(os.path.join(output_dir, "boat_removed.jpg"), boat_removed)
    
    if is_encroach(boat_removed, source="true", boat=there_boat):
        print("Encroachment Found")
        return True
    else:
        print("No Encroachment Found")
        return False


class VesselTracker:
    """
    Tracks vessels in a video stream using a YOLO model.
    """
    def __init__(self, video_input_path, output_path, frame_rate=30, model_path=None):
        self.video_input_path = video_input_path
        self.output_path = output_path
        self.frame_rate = frame_rate
        
        if model_path is None:
            # Use a relative path to the tracking model
            model_path = os.path.join(BASE_DIR, "tracking", "runs", "detect", "train", "weights", "best.pt")
        self.model = YOLO(model_path).to(device)
    
    def track_vessels(self):
        """
        Processes the input video, annotates vessel tracking, and writes the output video.
        """
        cap = cv2.VideoCapture(self.video_input_path)
        writer = imageio.get_writer(self.output_path, fps=self.frame_rate)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            result = self.model(frame)
            result_frame = result[0].plot()
            
            cv2.imshow("Vessel Tracking", result_frame)
            rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        writer.close()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Example paths - replace with actual paths when using
    image_path = os.path.join(BASE_DIR, "HIN", "input.jpg")
    video_path = os.path.join(BASE_DIR, "tracking", "sample_video.mp4")
    output_video = os.path.join(BASE_DIR, "tracking", "output_video.mp4")
    
    # Uncomment the feature you want to test
    
    # Boat classification
    # classifier = BoatClassifier()
    # classifier.classify_boats(image_path)
    
    # HIN detection
    # hin_detector = HINDetector()
    # hin_value = hin_detector.detect_hin(image_path)
    # print(f"Detected HIN: {hin_value}")
    
    # Land encroachment check
    # encroachment = check_land_encroachment(image_path)
    # print(f"Land encroachment detected: {encroachment}")
    
    # Vessel tracking
    # tracker = VesselTracker(video_path, output_video)
    # tracker.track_vessels()