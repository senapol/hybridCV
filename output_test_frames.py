import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Extracting every {frame_interval} frames")

    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame as image
            output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Example usage
# extract_frames(
#     "camera1.mp4", 
#     "training_frames/camera1",
#     frame_interval=60  # Extract 1 frame per second for 60fps video
# )

# cap = cv2.VideoCapture("images/shortlandscape.MOV", cv2.CAP_FFMPEG)

# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print(f"FPS: {fps}, Total frames: {total_frames}")

extract_frames('images/shortlandscape.MOV', 'test-frames/train/', frame_interval=40)
