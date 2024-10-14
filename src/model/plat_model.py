import numpy as np
import cv2
from ultralytics import YOLO
from src.config.config import config



# class PlateDetector:
#     def __init__(self, model):
#         self.model = model

#     def preprocess(self, image: np.ndarray) -> np.ndarray:
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
#         return image_bgr

#     def predict(self, image: np.ndarray):
#         preprocessed_image = self.preprocess(image)
#         results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
#         return results

#     def detect_plate(self, frame, is_save=False):
#         results = self.predict(frame)

#         if not results:
#             print("[PlateDetector] No plates detected.")
#             return []

#         bounding_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() if results else []
#         if not bounding_boxes:
#             return []

#         cropped_plates = self.get_cropped_plates(frame, bounding_boxes)

#         if is_save:
#             self.save_cropped_plate(cropped_plates)

#         return cropped_plates

#     def save_cropped_plate(self, cropped_plates):
#         """
#         Save the cropped plate regions as image files.
#         Args:
#             cropped_plates: List of cropped plate images.
#         """
#         import os
#         from datetime import datetime

#         if not os.path.exists('plate_saved'):
#             os.makedirs('plate_saved')

#         for i, cropped_plate in enumerate(cropped_plates):
#             if cropped_plate.size > 0:
#                 # Create a filename with the current timestamp
#                 timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
#                 filename = f'plate_saved/{timestamp}.jpg'

#                 # Save the cropped plate image
#                 cv2.imwrite(filename, cropped_plate)

    # def is_valid_cropped_plate(self, cropped_plate):
    #     """Check if the cropped plate meets the size requirements."""
    #     height, width = cropped_plate.shape[:2]

    #     if not (25 <= height <= 40):  # Height range around the average 31.65
    #         return False
    #     if not (80 <= width <= 120):  # Width range around the average 97.14
    #         return False

    #     if height >= width:
    #         return False

    #     # compare = abs(height - width)
    #     # if not (30 <= compare <= 120):
    #     #     return False

    #     return True

#     # def is_valid_cropped_plate(self, cropped_plate):
#     #     """Check if the cropped plate meets the size requirements."""
#     #     height, width = cropped_plate.shape[:2]
#     #     if height < 55 or width < 100:
#     #         return False
#     #     if height >= width:
#     #         return False
#     #     compare = abs(height - width)
#     #     if compare <= 100 or compare >= 400:
#     #         return False
#     #     return True

#     def get_cropped_plates(self, frame, boxes):
#         """
#         Extract cropped plate images based on bounding boxes.
#         Args:
#             frame: The original image/frame.
#             boxes: List of bounding boxes (each box is [x1, y1, x2, y2]).

#         Returns:
#             cropped_plates: List of cropped plate images.
#         """
#         height, width, _ = frame.shape
#         cropped_plates = []

#         for box in boxes:
#             x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]
#             cropped_plate = frame[y1:y2, x1:x2]

#             print("cropped_plate.shape: ", cropped_plate.shape)

#             if cropped_plate.size > 0:
#             # if cropped_plate.size > 0 and self.is_valid_cropped_plate(cropped_plate):
#                 cropped_plates.append(cropped_plate)

#         return cropped_plates


#     def draw_boxes(self, frame, boxes):
#         """
#         Draw bounding boxes for detected plates on the frame.
#         Args:
#             frame: The original image/frame.
#             boxes: List of bounding boxes to draw (each box is [x1, y1, x2, y2]).
#         """
#         height, width, _ = frame.shape

#         for box in boxes:
#             x1, y1, x2, y2 = [max(0, min(int(coord), width if i % 2 == 0 else height)) for i, coord in enumerate(box)]

#             color = (0, 255, 0)  # Green
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red

#         return frame

class PlateDetector:
    def __init__(self, plate_detection_model):
        # TODO refactor YOLO model
        self.model = plate_detection_model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # TODO image processing
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def predict(self, image: np.ndarray):
        preprocessed_image = self.preprocess(image)
        # results = self.model.predict(preprocessed_image, conf=0.25, device="cuda:0", verbose=False, classes=config.CLASS_PLAT_NAMES)
        results = self.model.predict(preprocessed_image, conf=0.3, device="cuda:0", verbose=False)
        return results

    def draw_boxes(self, frame, results):
        for box in results.boxes.xyxy.cpu():
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # center_x = (x1 + x2) // 2
            # center_y = (y1 + y2) // 2
            # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

    def filter_object(self, result, selected_class_names: dict):
        if len(result.boxes) == 0:
            return result

        indices = [
            i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
            if int(cls) in selected_class_names.keys()
        ]
        result.boxes = result.boxes[indices]

    def get_plat_image(self, image, results):
        plat = np.array([])
        for box in results.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(int, box)
            plat = image[max(y1, 0): min(y2, image.shape[0]), max(x1, 0): min(x2, image.shape[1])]
            
            if self.is_valid_cropped_plate(plat):
                return plat

        return np.array([])

    def is_valid_cropped_plate(self, cropped_plate):
        """Check if the cropped plate meets the size requirements."""
        height, width = cropped_plate.shape[:2]

        if not (25 <= height <= 40):  # Height range around the average 31.65
            return False
        if not (80 <= width <= 120):  # Width range around the average 97.14
            return False

        if height >= width:
            return False

        # Compare height and width differences
        # compare = abs(height - width)
        # if not (30 <= compare <= 120):
        #     return False

        return True