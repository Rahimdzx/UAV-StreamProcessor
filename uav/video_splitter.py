import cv2
import os
import argparse
import random

def extract_frames(video_path: str, output_dir: str) -> None:
    """
    Extracts every frame from the input video and saves them as JPEG images.
    :param video_path: Path to the input video file.
    :param output_dir: Directory where the frame images will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames into directory: {output_dir}")

def split_video_into_random_segments(video_path: str, output_dir: str, num_segments: int = 4) -> None:
    """
    Splits the input video into a specified number of segments of random (but contiguous) lengths.
    :param video_path: Path to the input video file.
    :param output_dir: Directory where the segmented video files will be stored.
    :param num_segments: The number of segments to split the video into.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure that there are enough frames to make splits. (Requires at least 5 frames.)
    if total_frames < num_segments + 1:
        raise ValueError("Not enough frames in the video to split into the requested number of segments.")

    # Randomly choose (num_segments - 1) boundaries between frame 1 and total_frames-1,
    # then sort them to produce segment boundaries.
    split_points = sorted(random.sample(range(1, total_frames), num_segments - 1))
    boundaries = [0] + split_points + [total_frames]
    
    print(f"Total frames: {total_frames}")
    print(f"Segment boundaries (frame indices): {boundaries}")

    # Iterate over segments and write out the frames
    for seg_index in range(num_segments):
        start_frame = boundaries[seg_index]
        end_frame = boundaries[seg_index + 1]
        segment_length = end_frame - start_frame
        
        segment_filename = os.path.join(output_dir, f"segment_{seg_index + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_filename, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Could not open writer for segment {seg_index + 1}")
            continue

        # Set the video to the correct starting frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written_frames = 0
        for _ in range(segment_length):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written_frames += 1
        writer.release()
        print(f"Segment {seg_index + 1}: Saved {written_frames} frames to {segment_filename}")

    cap.release()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility to extract individual frames from a video and split the video into 4 random segments."
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--frames_output_dir",
        type=str,
        required=True,
        help="Directory where extracted frames will be saved."
    )
    parser.add_argument(
        "--segments_output_dir",
        type=str,
        required=True,
        help="Directory where the video segments will be saved."
    )
    args = parser.parse_args()

    print("Starting frame extraction...")
    extract_frames(args.input_video, args.frames_output_dir)

    print("\nStarting video segmentation into random lengths...")
    split_video_into_random_segments(args.input_video, args.segments_output_dir)

if __name__ == "__main__":
    main()
