
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from ultralytics import YOLO

from custom_types import BBox, Frame

# Setup basic logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrameResizer:
    
    def __init__(self, target_resolution: int):
        if target_resolution <= 0:
            raise ValueError("Target resolution must be positive.")
        self._target_resolution = target_resolution
        self._target_size = (target_resolution, target_resolution)
        logging.info(f"FrameResizer initialized with target resolution: {target_resolution}x{target_resolution}")

    def __call__(self, frame: Frame) -> Frame:
       
        h, w, _ = frame.shape
        target_w, target_h = self._target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        try:
            resized_image = cv2.resize(frame.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            top_pad = (target_h - new_h) // 2
            bottom_pad = target_h - new_h - top_pad
            left_pad = (target_w - new_w) // 2
            right_pad = target_w - new_w - left_pad

            padded_image = cv2.copyMakeBorder(resized_image,
                                              top_pad, bottom_pad,
                                              left_pad, right_pad,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Black padding

            # Create a new Frame object for the result
            resized_frame = Frame(image=padded_image,
                                  frame_number=frame.frame_number,
                                  original_shape=frame.original_shape, # Keep original shape info
                                  fps=frame.fps)
            return resized_frame

        except Exception as e:
            logging.error(f"Error resizing frame #{frame.frame_number}: {e}")
            # Return original frame or raise error? Let's return original for now.
            # Consider adding specific error handling or raising.
            return frame


class ObjectDetector:
    
    # Default classes to detect if none specified, matching original request
    DEFAULT_CLASSES = ['car', 'person']
    DEFAULT_CONFIDENCE = 0.4

    def __init__(self, path_to_checkpoint: Union[str, Path],
                 target_classes: Optional[List[str]] = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE):
        
        try:
            logging.info(f"Loading YOLO model from: {path_to_checkpoint}")
            self.model = YOLO(str(path_to_checkpoint))
            logging.info("YOLO model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

        self.target_classes = target_classes if target_classes is not None else self.DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold

        # Get class IDs for faster lookup
        self.target_class_ids = [
            k for k, v in self.model.names.items() if v in self.target_classes
        ]
        logging.info(f"ObjectDetector initialized. Target classes: {self.target_classes} (IDs: {self.target_class_ids}), Confidence: {self.confidence_threshold}")


    def __call__(self, frame: Frame) -> List[BBox]:
        
        detected_boxes: List[BBox] = []
        try:
            # Perform inference; verbose=False suppresses console output
            results = self.model(frame.image, verbose=False, conf=self.confidence_threshold)

            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # Check if the detected class is one we are interested in
                    if cls_id in self.target_class_ids:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = self.model.names[cls_id]

                        # Create BBox object
                        bbox = BBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                                    confidence=conf, class_id=cls_id, class_name=class_name)
                        detected_boxes.append(bbox)

        except Exception as e:
            logging.error(f"Error during object detection on frame #{frame.frame_number}: {e}")
            # Return empty list on error

        # Optional: Draw boxes on the frame for debugging/visualization (modify if needed)
        # self._draw_boxes(frame.image, detected_boxes)

        return detected_boxes

    # Helper to draw boxes (optional)
    def _draw_boxes(self, image: np.ndarray, boxes: List[BBox]):
        
        for box in boxes:
            label = f"{box.class_name}: {box.confidence:.2f}"
            cv2.rectangle(image, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (box.x_min, box.y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class FrameSaver:
    
    DEFAULT_FPS = 25.0 # Default FPS if not provided by reader

    def __init__(self, path_to_target_mp4: Union[str, Path]):
        self.target_path = Path(path_to_target_mp4)
        # Ensure output directory exists
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer: Optional[cv2.VideoWriter] = None
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        self._fps: Optional[float] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        logging.info(f"FrameSaver initialized. Output file: {self.target_path}")

    def _initialize_writer(self, frame: Frame):
        
        try:
            self._frame_size = (frame.original_shape[1], frame.original_shape[0]) # Use original W, H
            # Prefer FPS from the frame/reader, otherwise use default
            self._fps = frame.fps if frame.fps is not None and frame.fps > 0 else self.DEFAULT_FPS
            self.writer = cv2.VideoWriter(str(self.target_path), self._fourcc, self._fps, self._frame_size)
            if not self.writer.isOpened():
                raise IOError("Failed to open VideoWriter")
            logging.info(f"FrameSaver: VideoWriter initialized. Size: {self._frame_size}, FPS: {self._fps}")
        except Exception as e:
            logging.error(f"Failed to initialize VideoWriter: {e}")
            self.writer = None # Ensure writer is None if init fails

    def __call__(self, frame: Frame) -> None:
        
        # Initialize writer on the first call
        if self.writer is None:
            self._initialize_writer(frame)
            # If initialization failed, don't try to write
            if self.writer is None:
                logging.warning(f"FrameSaver: Skipping frame #{frame.frame_number} due to writer initialization failure.")
                return

        # Ensure frame dimensions match writer dimensions (use original frame)
        if frame.original_shape[1] != self._frame_size[0] or frame.original_shape[0] != self._frame_size[1]:
             logging.warning(f"FrameSaver: Frame #{frame.frame_number} size mismatch "
                            f"({frame.original_shape[1]}x{frame.original_shape[0]}) vs "
                            f"writer ({self._frame_size[0]}x{self._frame_size[1]}). Skipping write.")
             # This shouldn't happen if we always use the original frame dimensions
             # If resizing happens before saving, logic needs adjustment.
             # Based on main.py, saver gets the *original* frame.
             return

        try:
            # IMPORTANT: Write the original image data, not potentially resized data
            # Assuming the 'frame' passed to saver is the original one from the reader
             original_image = frame.image # If frame obj always holds current image, need original
             # Let's assume the `Frame` object passed to the saver IS the original one.
             # The main loop should pass the frame from the reader to the saver.
            self.writer.write(frame.image)
        except Exception as e:
    logging.error(f"Error writing frame #{frame.frame_number} to {self.target_path}: {e}")

    def release(self):
        """Releases the VideoWriter resource."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()
            logging.info(f"FrameSaver released: {self.target_path}")
            self.writer = None

    def __del__(self):
        # Ensure release is called even if user forgets
        self.release()


class FrameSender:
    """Encodes and sends frames via HTTP POST."""
    POST_TIMEOUT = 10 # Seconds

    def __init__(self, target_url: str):
        if not target_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid target URL: {target_url}")
        self._target_url = target_url
        logging.info(f"FrameSender initialized. Target URL: {self._target_url}")

    def __call__(self, frame: Frame, bboxes: Optional[List[BBox]]):
        """
        Encodes the frame image as JPEG and sends it along with bbox data.
        :param frame: The Frame object (typically resized) to send.
        :param bboxes: List of detected BBox objects or None.
        """
        try:
            # 1. Encode the image
            is_success, buffer = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not is_success:
                logging.error(f"Failed to encode frame #{frame.frame_number} to JPEG.")
                return

            # 2. Prepare multipart/form-data payload
            files = {'frame': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

            # 3. Add bounding box data (e.g., as JSON string in a form field)
            bbox_data_list = []
            if bboxes:
                for bbox in bboxes:
                     bbox_data_list.append({
                         'class_name': bbox.class_name,
                         'class_id': bbox.class_id,
                         'confidence': round(bbox.confidence, 4),
                         'box': [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
                     })

            # Add frame number maybe
            metadata = {'frame_number': frame.frame_number}
            if bbox_data_list:
                metadata['detections'] = bbox_data_list

            # Convert metadata dict to JSON string if needed by the receiver
            # Or send as separate form fields. Sending as single JSON field is common.
            import json
            data = {'metadata': json.dumps(metadata)}

            # 4. Send POST request
            response = requests.post(self._target_url, files=files, data=data, timeout=self.POST_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            logging.debug(f"Frame #{frame.frame_number} sent successfully to {self._target_url}. Status: {response.status_code}")
            # Use debug level for successful sends to avoid flooding logs

        except requests.exceptions.Timeout:
             logging.warning(f"Timeout sending frame #{frame.frame_number} to {self._target_url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending frame #{frame.frame_number} to {self._target_url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during frame sending (frame #{frame.frame_number}): {e}")