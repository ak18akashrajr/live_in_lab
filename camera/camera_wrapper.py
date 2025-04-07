import cv2
import threading
import time


class Camera:
    def __init__(self, src=0):
        self.src = src  # Camera source (0 for default webcam)
        self.cap = cv2.VideoCapture(self.src)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self._update_frame, daemon=True)
            self.thread.start()
            print("[Camera] Capture thread started.")

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.01)  # Add small delay to reduce CPU load

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        print("[Camera] Capture thread stopped and camera released.")
