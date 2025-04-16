
import numpy as np
from typing import Optional, Tuple

class BBox:
    
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int,
                 confidence: float, class_id: int, class_name: str):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self):
        return (f"BBox(class='{self.class_name}', "
                f"coords=({self.x_min},{self.y_min},{self.x_max},{self.y_max}), "
                f"conf={self.confidence:.2f})")

class Frame:
    
    def __init__(self, image: np.ndarray, frame_number: int = -1,
                 original_shape: Optional[Tuple[int, int, int]] = None,
                 fps: Optional[float] = None):
        self.image: np.ndarray = image
        self.frame_number: int = frame_number
        # Store original shape if it's modified (e.g., by resizing)
        self.original_shape: Optional[Tuple[int, int, int]] = original_shape or image.shape
        # Store fps if available from the reader
        self.fps: Optional[float] = fps

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape