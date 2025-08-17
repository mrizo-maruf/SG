import cv2
import os
import argparse
import re

def frames_to_video(frames_dir, output_path, fps=30):
    # if output_path is a directory or has no extension, use output.mp4 inside it
    if os.path.isdir(output_path) or os.path.splitext(output_path)[1] == "":
        output_path = os.path.join(output_path, "output.mp4")

    # get all frames matching pattern
    frames = [f for f in os.listdir(frames_dir) if re.match(r"^frame\d+\.jpg$", f)]
    frames.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))  # sort by number
    print(frames)
    if not frames:
        print("❌ No frames found in", frames_dir)
        return

    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first_frame is None:
        print(f"❌ Unable to read first frame: {frames[0]}")
        return
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in frames:
        img_path = os.path.join(frames_dir, f)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable frame {f}")
            continue
        out.write(frame)

    out.release()
    print(f"✅ Video saved at {output_path}")

if __name__ == "__main__":
    frames_dir = "/workspace/isaaclab/IsaacSimData/warehouse/results"
    output_path = "/workspace/isaaclab/IsaacSimData/warehouse"  # can be dir or file
    fps=  30
    frames_to_video(frames_dir, output_path, fps)
