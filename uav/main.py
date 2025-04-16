# main.py
import argparse
import logging
import time
from pathlib import Path

# Assuming custom_types, frame_reader, frame_processing are in the same directory or accessible
from frame_reader import SourceReader, StreamReader # Import StreamReader to check instance type
from frame_processing import FrameResizer, FrameSaver, FrameSender, ObjectDetector

# Setup logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(
        "Run BVS Video Processing Pipeline.",
        add_help=True)
    parser.add_argument(
        "--path_to_source",
        type=str,
        required=True,
        help="Path to the input video file, image directory, or RTSP stream URL."
    )
    parser.add_argument(
        "--target_resolution",
        default=640,
        type=int,
        help="Target square resolution for processed keyframes (e.g., 640 for 640x640)."
    )
    parser.add_argument(
        "--path_to_output_mp4",
        type=str,
        required=True,
        help="Path to save the recorded input stream as an MP4 file."
    )
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        default='yolov8n.pt', # Default to nano version
        help="Path or name of the YOLO model checkpoint (e.g., 'yolov8n.pt', '/path/to/your/best.pt')."
    )
    parser.add_argument(
        "--target_url",
        type=str,
        required=True,
        help="URL endpoint to send processed keyframes with bounding box predictions via POST."
    )
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=30,  # Process every 30th frame by default
        help="Interval for keyframe extraction (process every N-th frame)."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for object detection."
    )
    parser.add_argument(
        '--target_classes',
        nargs='+', # Allows multiple class names
        default=['car', 'person'], # Default classes from original task description
        help="List of object classes to detect (e.g., --target_classes person car truck)."
    )

    args = parser.parse_args()
    if args.keyframe_interval <= 0:
        parser.error("--keyframe_interval must be a positive integer.")
    return args


if __name__ == "__main__":
    args = get_args()
    logger.info("Starting video processing pipeline...")
    logger.info(f"Input source: {args.path_to_source}")
    logger.info(f"Output recording: {args.path_to_output_mp4}")
    logger.info(f"Keyframe interval: Process 1 out of every {args.keyframe_interval} frames")
    logger.info(f"Target resolution for keyframes: {args.target_resolution}x{args.target_resolution}")
    logger.info(f"Object detection model: {args.path_to_checkpoint}")
    logger.info(f"Detection confidence: {args.confidence}")
    logger.info(f"Target classes: {args.target_classes}")
    logger.info(f"POST endpoint: {args.target_url}")

    # Use try...finally to ensure resources are released
    frame_reader = None
    saver = None
    start_time = time.time()
    total_frame_count = 0 # Counter for all frames read/saved
    processed_keyframe_count = 0 # Counter for frames actually processed

    try:
        # Initialize components
        # Use a context manager for the reader if possible (requires reader implementing __enter__/__exit__)
        # If not, rely on finally block. SourceReader implements it now.
        frame_reader = SourceReader(args.path_to_source)

        detector = ObjectDetector(args.path_to_checkpoint,
                                  target_classes=args.target_classes,
                                  confidence_threshold=args.confidence)
        resizer = FrameResizer(args.target_resolution)
        saver = FrameSaver(args.path_to_output_mp4) # Saver for the original stream
        sender = FrameSender(args.target_url) # Sender for processed keyframes

        logger.info("Initialization complete. Starting frame processing loop...")

        # Main processing loop
        for frame in frame_reader:
            try:
                # 1. Save the original frame unconditionally
                # FrameSaver handles its own initialization on the first frame
                saver(frame)

                # 2. Check if the current frame is a keyframe based on the interval
                if total_frame_count % args.keyframe_interval == 0:
                    processed_keyframe_count += 1
                    logger.info(f"---> Processing keyframe #{total_frame_count} (Keyframe #{processed_keyframe_count})")

                    # 2.1 Resize the key frame
                    # Pass the current frame (which is the keyframe) to the resizer
                    res_frame = resizer(frame)

                    # 2.2 Detect objects on the *resized* key frame
                    bboxes = detector(res_frame) # Returns List[BBox]
                    if bboxes:
                         logger.info(f"     Detected {len(bboxes)} objects: {[b.class_name for b in bboxes]}")
                    else:
                         logger.info("     No target objects detected.")


                    # 2.3 Send the *resized* key frame and its detections
                    # Pass the resized frame (res_frame) and the list of bboxes
                    sender(res_frame, bboxes)
                else:
                    # Frame is not a keyframe, skip processing (resizing, detection, sending)
                    # Logging this might be too verbose, remove if not needed
                    # logger.debug(f"Skipping frame #{total_frame_count} (not a keyframe)")
                    pass

                total_frame_count += 1

                # Log progress periodically based on total frames processed
                if total_frame_count % 100 == 0:
                     logger.info(f"Processed {total_frame_count} total frames (saved). "
                                 f"{processed_keyframe_count} keyframes sent for processing.")

                # (Optional) Add a small sleep for RTSP streams if processing is too fast
                # Check the underlying reader type within SourceReader
                # if isinstance(frame_reader.reader, StreamReader):
                #    time.sleep(0.01) # Adjust as needed

            except StopIteration:
                logger.info("StopIteration received, ending loop gracefully.")
                break # Exit the loop cleanly if iterator signals end
            except Exception as loop_error:
                logger.error(f"Error processing frame #{total_frame_count}: {loop_error}", exc_info=False) # exc_info=True for traceback
                # Decide whether to continue or break on error
                # continue # Or break

        logger.info("Finished processing stream loop.")

    except Exception as e:
        # Catch errors during initialization or other unexpected issues
        logger.error(f"An error occurred outside the main processing loop: {e}", exc_info=True)
    finally:
        # Ensure resources are released regardless of how the loop/try block exited
        if saver:
            logger.info("Releasing FrameSaver...")
            saver.release()
        if frame_reader:
            logger.info("Releasing FrameReader...")
            frame_reader.release() # SourceReader delegates release

        end_time = time.time()
        logger.info(f"Pipeline finished.")
        logger.info(f"Total frames read/saved: {total_frame_count}")
        logger.info(f"Total keyframes processed/sent: {processed_keyframe_count}")
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")