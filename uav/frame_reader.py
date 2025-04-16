import cv2
import os
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Union
import logging

from custom_types import Frame

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrameReader(ABC):
    """Abstract base class for reading frames from a source."""
    def __init__(self):
        self._fps: Optional[float] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    @abstractmethod
    def __next__(self) -> Frame:
        """Return the next frame."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """Return the iterator object."""
        pass

    @abstractmethod
    def release(self):
        """Release any resources held by the reader."""
        pass

    @property
    def fps(self) -> Optional[float]:
        return self._fps

    @property
    def width(self) -> Optional[int]:
        return self._width

    @property
    def height(self) -> Optional[int]:
        return self._height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ImageReader(FrameReader):
    """Reads frames from a directory of ordered image files."""
    def __init__(self, path_to_source: Path):
        super().__init__()
        if not path_to_source.is_dir():
            raise ValueError(f"ImageReader expects a directory, got {path_to_source}")

        self.image_files: List[Path] = sorted(
            [p for p in path_to_source.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')],
            key=lambda x: int(re.search(r'\d+', x.stem).group()) if re.search(r'\d+', x.stem) else float('inf')
            # Attempt to sort numerically based on digits in filename
        )

        if not self.image_files:
            raise FileNotFoundError(f"No supported image files found in {path_to_source}")

        self._index = 0
        self._frame_count = 0

        # Try to get dimensions from the first image
        try:
            first_image = cv2.imread(str(self.image_files[0]))
            if first_image is None:
                 raise IOError(f"Could not read the first image: {self.image_files[0]}")
            self._height, self._width, _ = first_image.shape
            # FPS is not well-defined for image sequences, use a default or None
            self._fps = None # Or set a default like 25 or 30 if needed downstream
            logging.info(f"ImageReader: Found {len(self.image_files)} images. Dimensions: {self._width}x{self._height}. FPS assumed: {self._fps}")
        except Exception as e:
             logging.error(f"Failed to read first image to get dimensions: {e}")
             raise

    def __iter__(self) -> Iterator[Frame]:
        self._index = 0
        self._frame_count = 0
        return self

    def __next__(self) -> Frame:
        if self._index >= len(self.image_files):
            raise StopIteration

        file_path = self.image_files[self._index]
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                raise IOError(f"Failed to read image file: {file_path}")

            frame = Frame(image=image, frame_number=self._frame_count, fps=self._fps)
            self._index += 1
            self._frame_count += 1
            return frame
        except Exception as e:
            logging.error(f"Error reading or processing image {file_path}: {e}")
            # Decide whether to stop or skip
            raise StopIteration # Stop on error

    def release(self):
        logging.info("ImageReader released (no resources to free).")
        pass # Nothing to release for image files


class VideoReader(FrameReader):
    """Reads frames from a video file."""
    def __init__(self, path_to_source: Path):
        super().__init__()
        if not path_to_source.is_file():
            raise FileNotFoundError(f"Video file not found: {path_to_source}")

        self.source_path = str(path_to_source)
        self.cap = cv2.VideoCapture(self.source_path)

        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {self.source_path}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = 0
        logging.info(f"VideoReader: Opened {self.source_path}. Resolution: {self._width}x{self._height}, FPS: {self._fps:.2f}")

    def __iter__(self) -> Iterator[Frame]:
        # Reset stream to the beginning if possible (might need re-opening for some cases)
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Safer to reopen if needed multiple times
        # For simplicity, assume single iteration or handle outside
        self._frame_count = 0 # Reset frame counter for new iteration
        return self

    def __next__(self) -> Frame:
        ret, image = self.cap.read()
        if not ret:
            self.release() # Release when stream ends
            raise StopIteration

        frame = Frame(image=image, frame_number=self._frame_count, fps=self._fps)
        self._frame_count += 1
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info(f"VideoReader released: {self.source_path}")


class StreamReader(FrameReader):
    """Reads frames from an RTSP stream."""
    def __init__(self, stream_url: str):
        super().__init__()
        self.source_url = stream_url
        # Add environment variables for potential RTSP latency tuning if needed
        # os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp' # Or tcp
        self.cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise IOError(f"Could not open RTSP stream: {self.source_url}")

        # Try to get properties, might be unreliable for RTSP initially
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._fps == 0:
            logging.warning("StreamReader: Could not reliably determine FPS from stream. Defaulting might be needed downstream.")
            self._fps = None # Indicate unknown FPS
        logging.info(f"StreamReader: Opened {self.source_url}. Resolution: {self._width}x{self._height}, FPS: {self._fps}")
        self._frame_count = 0

    def __iter__(self) -> Iterator[Frame]:
        self._frame_count = 0 # Reset frame counter
        return self

    def __next__(self) -> Frame:
        ret, image = self.cap.read()
        if not ret:
            logging.warning(f"StreamReader: Failed to grab frame from {self.source_url}. Stream might have ended or timed out.")
            # Don't release immediately on first failure, could be temporary
            # Consider adding retry logic or a timeout mechanism
            raise StopIteration # Or handle error differently

        frame = Frame(image=image, frame_number=self._frame_count, fps=self._fps)
        self._frame_count += 1
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info(f"StreamReader released: {self.source_url}")


class SourceReader(FrameReader):
    """Factory class to create the appropriate FrameReader based on the source path."""
    def __init__(self, path_to_source: Union[str, Path]):
        super().__init__() # Initialize base properties
        source_str = str(path_to_source)

        try:
            if source_str.lower().startswith("rtsp://"):
                logging.info(f"Source detected as RTSP stream: {source_str}")
                self.reader: FrameReader = StreamReader(source_str)
            else:
                source_path = Path(path_to_source)
                if source_path.is_dir():
                    logging.info(f"Source detected as image directory: {source_path}")
                    self.reader = ImageReader(source_path)
                elif source_path.is_file():
                    logging.info(f"Source detected as video file: {source_path}")
                    self.reader = VideoReader(source_path)
                else:
                    raise ValueError(f"Source path is not a valid file, directory, or RTSP URL: {source_path}")

            # Propagate properties from the specific reader
            self._fps = self.reader.fps
            self._width = self.reader.width
            self._height = self.reader.height

        except Exception as e:
            logging.error(f"Error initializing SourceReader for '{path_to_source}': {e}")
            raise # Re-raise the exception after logging

    def __next__(self) -> Frame:
        try:
            return self.reader.__next__()
        except StopIteration:
            raise # Propagate StopIteration correctly
        except Exception as e:
            logging.error(f"Error getting next frame from reader: {e}")
            self.release() # Attempt cleanup on error
            raise StopIteration # Stop iteration on error

    def __iter__(self) -> Iterator[Frame]:
        return self.reader.__iter__() # Delegate iteration

    def release(self):
        """Release resources of the underlying reader."""
        if hasattr(self, 'reader') and self.reader:
            self.reader.release()