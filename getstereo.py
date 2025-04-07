import pandas as pd
import os

def create_stereo_detections(csv_path, output_path=None):
    """
    Merge detections from camera 1 and camera 2 into single stereo detection rows,
    with columns for both cameras' coordinates.
    """
    print(f"Loading detections from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total detections loaded: {len(df)}")
    
    frames_camera1 = set(df[df['camera'] == 1]['frame_no'])
    frames_camera2 = set(df[df['camera'] == 2]['frame_no'])
    common_frames = frames_camera1.intersection(frames_camera2)
    
    print(f"Found {len(common_frames)} frames with detections from both cameras")
    
    stereo_rows = []
    
    for frame_no in sorted(common_frames):
        frame_detections = df[df['frame_no'] == frame_no]
        cam1_detections = frame_detections[frame_detections['camera'] == 1]
        cam2_detections = frame_detections[frame_detections['camera'] == 2]
        
        if len(cam1_detections) > 1:
            cam1_detections = cam1_detections.sort_values('confidence', ascending=False).iloc[:1]
        
        if len(cam2_detections) > 1:
            cam2_detections = cam2_detections.sort_values('confidence', ascending=False).iloc[:1]
        
        if not cam1_detections.empty and not cam2_detections.empty:
            cam1_row = cam1_detections.iloc[0]
            cam2_row = cam2_detections.iloc[0]
            
            stereo_row = {
                'frame_no': frame_no,
                'A1_x': cam1_row['x'],
                'A1_y': cam1_row['y'],
                'A1_confidence': cam1_row['confidence'],
                'A1_method': cam1_row['detection_method'],
                'A2_x': cam2_row['x'],
                'A2_y': cam2_row['y'],
                'A2_confidence': cam2_row['confidence'],
                'A2_method': cam2_row['detection_method']
            }
            
            stereo_rows.append(stereo_row)
    
    stereo_df = pd.DataFrame(stereo_rows)
    
    print(f"Created {len(stereo_df)} stereo detection pairs")
    
    if output_path:
        stereo_df.to_csv(output_path, index=False)
        print(f"Saved stereo detections to '{output_path}'")
    
    return stereo_df

if __name__ == "__main__":
    input_csv = 'output/detections_A1_A2-2s.csv'
    
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, 'stereo_detections.csv')
    
    stereo_df = create_stereo_detections(input_csv, output_csv)
    
    if not stereo_df.empty:
        print("\nSample stereo detections:")
        print(stereo_df.head())
