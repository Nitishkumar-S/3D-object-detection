import cv2
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Stitch images into a video')
    parser.add_argument('image_dir', help='Directory containing the .png frames')
    parser.add_argument('--output', default='demo_video.mp4', help='Name of the output video file')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second (nuScenes is 12Hz)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get all PNG files and sort them numerically/alphabetically
    images = [img for img in os.listdir(args.image_dir) if img.endswith(".png")]
    images.sort() 

    if not images:
        print(f"Error: No images found in {args.image_dir}")
        return

    # Determine the width and height from the first image
    first_image_path = os.path.join(args.image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the VideoWriter (MP4V is a standard codec that works on Windows/Mac)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    print(f"Stitching {len(images)} frames into {args.output}...")

    for image in tqdm(images):
        img_path = os.path.join(args.image_dir, image)
        frame = cv2.imread(img_path)
        
        # In case some images are slightly different sizes, resize them to match the first
        if (frame.shape[1] != width) or (frame.shape[0] != height):
            frame = cv2.resize(frame, (width, height))
            
        video.write(frame)

    video.release()
    print(f"\nSuccess! Video saved as: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()