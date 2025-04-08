import pandas as pd
import os

def create_stereo_detections(df, stereo='A'):
    """
    Merge detections from camera 1 and camera 2 into single stereo detection rows,
    with columns for both cameras' coordinates.
    """
    if stereo == 'A':
        cameras = (1,2)
    elif stereo == 'B':
        cameras = (3,4)
    else:
        print('Stereo should be A or B')
        exit()
    
    print(f'Generating stereo detections csv for cameras {cameras}')
    
    df = df[df['camera'].isin(cameras)]
    print(f"Total {stereo} detections loaded: {len(df)}")
    
    frames_cam1 = set(df[df['camera'] == cameras[0]]['frame_no'])
    frames_cam2 = set(df[df['camera'] == cameras[1]]['frame_no'])
    common_frames = frames_cam1.intersection(frames_cam2)
    
    print(f"Found {len(common_frames)} frames with detections from both {stereo} cameras")
    
    stereo_rows = []
    
    for frame_no in sorted(common_frames):
        frame_detections = df[df['frame_no'] == frame_no]
        cam1_detections = frame_detections[frame_detections['camera'] == cameras[0]]
        cam2_detections = frame_detections[frame_detections['camera'] == cameras[1]]
        
        if len(cam1_detections) > 1:
            cam1_detections = cam1_detections.sort_values('confidence', ascending=False).iloc[:1]
        
        if len(cam2_detections) > 1:
            cam2_detections = cam2_detections.sort_values('confidence', ascending=False).iloc[:1]
        
        if not cam1_detections.empty and not cam2_detections.empty:
            cam1_row = cam1_detections.iloc[0]
            cam2_row = cam2_detections.iloc[0]
            
            stereo_row = {
                'frame_no': frame_no,
                f'{stereo}1_x': cam1_row['x'],
                f'{stereo}1_y': cam1_row['y'],
                f'{stereo}1_confidence': cam1_row['confidence'],
                f'{stereo}1_method': cam1_row['detection_method'],
                f'{stereo}2_x': cam2_row['x'],
                f'{stereo}2_y': cam2_row['y'],
                f'{stereo}2_confidence': cam2_row['confidence'],
                f'{stereo}2_method': cam2_row['detection_method']
            }
            
            stereo_rows.append(stereo_row)
    
    stereo_df = pd.DataFrame(stereo_rows)
    
    print(f"Created {len(stereo_df)} {stereo} stereo detection pairs")
    
    return stereo_df
