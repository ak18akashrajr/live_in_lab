from ultralytics import YOLO
import torch
from camera import Camera
import cv2 as cv

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Model runs of ", device.type)


class DisasterManagement(Camera):
    def __init__(self, person_count_model, water_level_model=""):
        super().__init__()
        self.person_count_model = YOLO(person_count_model).to(device)
        # self.waterlevel_model = YOLO(water_level_model).to(device)

    def get_json_data(self):
        img = self.get_frame()
        # img = cv.imsave("test.jpg", img)
        presons = self.person_count_model(img, classes=[0])

        return {
            "person_count": len(presons[0]),
            "water_level": 0,
        }
