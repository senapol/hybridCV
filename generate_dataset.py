import cv2 as cv
from ultralytics import YOLO
from collections import deque
import numpy as np
import argparse
import pandas as pd
import os
from pathlib import Path
import shutil

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")

def init_lk(state, frame, centre):
    """Initializes optical flow state."""
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    state['prev_grey'] = grey
    state['lk_age'] = 0
    state['lk_pts'] = np.array([[centre]], dtype=np.float32)
    print(f"Initializing LK for camera {state['camera_id']} at {centre}")

def process_frame(frame, state, frame_no, model):
    """Processes one frame: uses YOLO if possible, otherwise optical flow fallback."""
    state['frame_count'] += 1
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # TODO: pre-process the frame
    
    results = model(frame)[0]
    detection = None

    if len(results.boxes) > 0 and results.boxes[0].conf[0].item() > 0.35:
        # Use highest confidence detection
        max_conf = max(results.boxes, key=lambda box: float(box.conf[0].item()))
        x1, y1, x2, y2 = max_conf.xyxy[0].cpu().numpy()
        centre = (int((x1 + x2) // 2), int((y1 + y2) // 2))
        conf = max_conf.conf[0].item()
        detection = {
            'centre': centre,
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': conf,
            'detection_method': 'YOLO',
            'frame_no': frame_no,
            'timestamp': state['cap'].get(cv.CAP_PROP_POS_MSEC)
        }
        state['pts'].appendleft((centre, 'YOLO'))
        init_lk(state, frame, centre)
        state['successful_frames'] += 1

    elif state['prev_grey'] is not None and state['lk_age'] < state['lk_max'] and state['lk_pts'] is not None:
        state['lk_age'] += 1
        new_points, status, error = cv.calcOpticalFlowPyrLK(
            state['prev_grey'], grey, state['lk_pts'], None, **state['lk_params'])
        print(f"Camera {state['camera_id']}: Optical flow fallback")
        if new_points is not None and status[0][0] == 1:
            x, y = new_points[0][0]
            centre = (int(x), int(y))
            state['pts'].appendleft((centre, 'LK'))
            state['prev_grey'] = grey
            state['lk_pts'] = new_points
            estimated_conf = max(0.2, 0.5 - (state['lk_age'] / 10))
            if state['lk_age'] > state['lk_max']:
                estimated_conf = 0.1
            detection = {
                'centre': centre,
                'bbox': None,  # No bounding box from LK; we will use a fixed size box later
                'confidence': estimated_conf,
                'detection_method': 'LK',
                'frame_no': frame_no,
                'timestamp': state['cap'].get(cv.CAP_PROP_POS_MSEC)
            }
            state['successful_frames'] += 1
        else:
            state['lk_pts'] = None
            print(f"Camera {state['camera_id']}: Optical flow lost")
    else:
        print(f"Camera {state['camera_id']}: No ball detected")

    return detection

def draw_annotation(frame, detection):
    """Draws a bounding box on the frame if a detection exists."""
    annotated = frame.copy()
    if detection is not None:
        if detection['bbox'] is not None:
            x1, y1, x2, y2 = detection['bbox']
            cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Fallback: draw a fixed-size box around the centre
            centre = detection['centre']
            box_size = 30
            x, y = centre
            cv.rectangle(annotated, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)
    return annotated

def process_videos(
  video1_path, 
  video2_path, 
  start_minute,
  end_minute,
  detected_frames_folder, 
  annotated_frames_folder, 
  csv_path, 
  model_path='models/last2.pt'
):
    # create output folders 
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(detected_frames_folder, exist_ok=True)
    os.makedirs(annotated_frames_folder, exist_ok=True)
    
    cap1 = cv.VideoCapture(video1_path)
    cap2 = cv.VideoCapture(video2_path)
    
    start_ms = start_minute * 60 * 1000
    end_ms = end_minute * 60 * 1000
    
    # set the video to the start time
    cap1.set(cv.CAP_PROP_POS_MSEC, start_ms)
    cap2.set(cv.CAP_PROP_POS_MSEC, start_ms)
  
    model = YOLO(model_path)

    # Initialize per-camera state dictionaries
    state1 = {
        'camera_id': 1,
        'cap': cap1,
        'frame_count': 0,
        'successful_frames': 0,
        'pts': deque(maxlen=64),
        'prev_grey': None,
        'lk_age': 0,
        'lk_max': 30,
        'lk_pts': None,
        'lk_params': dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))
    }
    state2 = {
        'camera_id': 2,
        'cap': cap2,
        'frame_count': 0,
        'successful_frames': 0,
        'pts': deque(maxlen=64),
        'prev_grey': None,
        'lk_age': 0,
        'lk_max': 30,
        'lk_pts': None,
        'lk_params': dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))
    }

    csv_data = []

    
    detection_count = 0
    while True:
        if detection_count > 10:
            print(f"Detection count: {detection_count}")
            break

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
          
        # Get current timestamp
        current_ms = cap1.get(cv.CAP_PROP_POS_MSEC)
        
        # Break if we've passed the end time
        if current_ms > end_ms:
            break

        frame_no = int(cap1.get(cv.CAP_PROP_POS_FRAMES))
        detection1 = process_frame(frame1, state1, frame_no, model)
        detection2 = process_frame(frame2, state2, frame_no, model)
        timestamp = cap1.get(cv.CAP_PROP_POS_MSEC)
        
        # First check if either detection is None
        if detection1 is None or detection2 is None:
            print(f"Detection is None for frame {frame_no}")
            continue
            
        # Then check the centres
        if detection1['centre'] is None or detection2['centre'] is None:
            print(f"Centre is None for frame {frame_no}")
            continue

        # Create subfolder for synchronized frames
        detected_subfolder = os.path.join(detected_frames_folder, f"frame_{frame_no}")
        os.makedirs(detected_subfolder, exist_ok=True)
        cv.imwrite(os.path.join(detected_subfolder, "cam1.png"), frame1)
        cv.imwrite(os.path.join(detected_subfolder, "cam2.png"), frame2)

        # Create annotated frames (with box drawn)
        ann_frame1 = draw_annotation(frame1, detection1)
        ann_frame2 = draw_annotation(frame2, detection2)
        annotated_subfolder = os.path.join(annotated_frames_folder, f"frame_{frame_no}")
        
        os.makedirs(annotated_subfolder, exist_ok=True)
        cv.imwrite(os.path.join(annotated_subfolder, "cam1.png"), ann_frame1)
        cv.imwrite(os.path.join(annotated_subfolder, "cam2.png"), ann_frame2)
        

        # Store detection info with same frame number for both cameras
        csv_data.append({
            'frame_number': frame_no,
            'timestamp': timestamp,
            'cam1_centre_x': detection1['centre'][0] if detection1 else None,
            'cam1_centre_y': detection1['centre'][1] if detection1 else None,
            'cam1_confidence': detection1['confidence'] if detection1 else None,
            'cam1_method': detection1['detection_method'] if detection1 else None,
            'cam2_centre_x': detection2['centre'][0] if detection2 else None,
            'cam2_centre_y': detection2['centre'][1] if detection2 else None,
            'cam2_confidence': detection2['confidence'] if detection2 else None,
            'cam2_method': detection2['detection_method'] if detection2 else None,
        })

        # # Optional: display annotated frames in separate windows
        # cv.imshow("Camera 1", ann_frame1)
        # cv.imshow("Camera 2", ann_frame2)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        detection_count += 1

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

    # Save CSV data
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print("Processing complete. CSV saved.")

def main():
    root = find_project_root()
    start_minute = 6
    end_minute = 8
  
    video1_path = f"{root}/videos/cam1.MOV"
    video2_path = f"{root}/videos/cam2.mp4"
    detected_frames_folder = f"{root}/output/detected_frames"
    annotated_frames_folder = f"{root}/output/annotated_frames"
    csv_path = f"{root}/output/detections.csv"
    model_path = f"{root}/models/last2.pt"
    
    # if csv exists, delete it
    if os.path.exists(csv_path):
        os.remove(csv_path)
      
    # if detected_frames_folder exists, delete it
    if os.path.exists(detected_frames_folder):
        shutil.rmtree(detected_frames_folder)
    
    # if annotated_frames_folder exists, delete it
    if os.path.exists(annotated_frames_folder):
        shutil.rmtree(annotated_frames_folder)
    
    process_videos(
      video1_path, 
      video2_path,
      start_minute,
      end_minute,
      detected_frames_folder, 
      annotated_frames_folder, 
      csv_path, 
      model_path
    )

if __name__ == '__main__':
    main()
