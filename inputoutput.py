import cv2 as cv
from dataclasses import dataclass
from ultralytics import YOLO
import time
from typing import Optional
from collections import deque
import numpy as np
import argparse as ap
import pandas as pd
import os
import csv
from getstereo import create_stereo_detections as ss_csv
from concurrent.futures import ThreadPoolExecutor

parser = ap.ArgumentParser()
parser.add_argument('--video1', type=str, help='Path to video file', default='images/A1-s.mov')
parser.add_argument('--video2', type=str, help='Path to video file', default='images/A2-s2.mov')
parser.add_argument('--video3', type=str, help='Path to video file', default='images/B1-g.mov')
parser.add_argument('--video4', type=str, help='Path to video file', default='images/B2-g.mov')
parser.add_argument('--model', type=str, help='Path to model file', default='models/last3.pt')
parser.add_argument('--start', type=int, help='Starting frame number', default=1020)
parser.add_argument('--end', type=int, help='Ending frame number', default=1680)
parser.add_argument('--save_output', type=bool, help='Save output to CSV', default=True)
parser.add_argument('--save_video', type=bool, help='Save output as video', default=True)
args = parser.parse_args()

frame_range = (args.start, args.end)
id = int(time.time()-1744060000)
output_A_csv = f'output/stereo_A_detections_{id}.csv'
output_B_csv = f'output/stereo_B_detections_{id}.csv'

videos = {
    1: args.video1,
    2: args.video2,
    3: args.video3,
    4: args.video4
}

@dataclass
class BallDetection:
    centre: tuple
    timestamp: float
    frame_no: int
    frame_count: int
    confidence: float
    detection_method: str # BG, YOLO, HSV, LK, Hough
    camera: int

class CameraProcessor:
    '''Processor for each camera (A1,A2,B1,B2)
    run() processes video frame by frame according to frame_range
    creates list of all detections'''
    def __init__(self, camera_id: int, video_path: str, model_path: str = 'models/last2.pt', output_dir='output/matching_frames', save_video=False):
        self.camera_id = camera_id
        self.csv_output = output_dir
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.frame_count = 0
        self.successful_frames = 0
        self.start_time = time.time()
        self.pts = deque(maxlen=64)
        self.consec_no_detection = 0
        self.max_consec_no_detection = 0
        self.detection_gaps = []
        self.save_video = save_video

        # background sub
        self.backSub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)
        self.prev_frame = None
        self.prev_Mt = None

        # lk optical flow
        self.prev_grey = None
        self.lk_age = 0
        self.lk_max = 10
        self.lk_pts = None
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))

        #properties
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Video writer setup
        if self.save_video:
            os.makedirs('output/videos', exist_ok=True)
            self.video_filename = f'output/videos/camera_{camera_id}_tracking_{id}.mp4'
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv.VideoWriter(self.video_filename, fourcc, self.fps, (self.width, self.height))
            print(f'Video will be saved to {self.video_filename}')
        else:
            self.video_writer = None

        print(f'Camera {camera_id} initialized with WxH {self.width}x{self.height} @ {self.fps}fps')

    def init_lk(self, frame, centre):
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.prev_grey = grey
        self.lk_age = 0
        self.lk_pts = np.array([[centre]], dtype=np.float32)
        print(f'Initialising LK at {centre}')

    def detect_bgsub(self, frame, grey):
        fgMask = self.backSub.apply(frame)
        _, Mb = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)

        kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        dilated = cv.dilate(Mb, kernel_dilate, iterations=2)
        eroded = cv.erode(dilated, kernel_erode, iterations=1)

        contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        M2b = np.zeros_like(eroded)

        for contour in contours:
            area = cv.contourArea(contour)

            if 10 < area < 500:
                cv.drawContours(M2b, [contour], -1, 255, -1)
            
        Md = np.zeros_like(grey)
        if self.prev_frame is not None:
            frame_diff = cv.absdiff(grey, self.prev_frame)
            _, Md = cv.threshold(frame_diff, 25, 255, cv.THRESH_BINARY)
            
        self.prev_frame = grey.copy()
        Mt = cv.bitwise_and(M2b, Md)

        Mpre = np.ones_like(Mt) * 255
        if self.prev_Mt is not None:
            prev_Mt_dilated = cv.dilate(self.prev_Mt, kernel_dilate, iterations=1)
            Mpre = cv.bitwise_not(prev_Mt_dilated)

        Mf = cv.bitwise_and(Mt, Mpre)

        self.prev_Mt = Mt.copy()
        ball_contours, _ = cv.findContours(Mf, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        ball_pt = None
        if ball_contours:
            best_circularity = 0
            best_contour = None
            for contour in ball_contours:
                area = cv.contourArea(contour)
                if area < 5:
                    continue
                perimeter = cv.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter ** 2))
                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_contour = contour
            if best_contour is not None and best_circularity > 0.6:
                print(f'CIRCULARITY: ')
                M = cv.moments(best_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    ball_pt = (cx, cy)

                    confidence = best_circularity
                    return ball_pt, confidence
        return None, 0
    
    def successful(self, pt, method, frame, conf):
        self.pts.appendleft((pt, method))
        self.successful_frames += 1
        if method != 'LK':        
            self.init_lk(frame, pt)
            
        if self.consec_no_detection > 0:
            self.detection_gaps.append(self.consec_no_detection)
            self.consec_no_detection = 0
    
        frame_dir = f'{self.csv_output}/frame_{self.frame_count}'
        os.makedirs(frame_dir, exist_ok=True)

        frame_c = frame.copy()
        cv.circle(frame_c, pt, 15, (0, 0, 255), 5)
        # cv.putText(frame, f'{method}: {conf:.2f}', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(frame_c, f'{method}: {conf:.2f}', (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # save frame
        csv_dir = f'{frame_dir}/{self.camera_id}_{self.frame_count}.jpg'
        try:
            cv.imwrite(csv_dir, frame_c)
            print(f'Saved frame to {csv_dir}')
        except Exception as e:
            print(f'Error saving frame: {e}')

    def process_frame(self, frame, timestamp: float, frame_no: int) -> Optional[BallDetection]:

        self.frame_count += 1
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        bg_pt, bg_conf = self.detect_bgsub(frame, grey)

        results = self.model(frame)[0]
        yolo_pt = None
        yolo_conf = 0.0

        if len(results.boxes) > 0 and results.boxes[0].conf[0].item() > 0.38:
            # get highest confidence detection
            max_conf = max(results.boxes, key=lambda box: float(box.conf[0].item()))
            x1, y1, x2, y2 = max_conf.xyxy[0].cpu().numpy()
            yolo_pt = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            yolo_conf = max_conf.conf[0].item()

        final_pt = None
        final_conf = 0.0
        method = None

        if bg_pt is not None and yolo_pt is not None:

            dist = np.sqrt((bg_pt[0] - yolo_pt[0])**2 + (bg_pt[1] - yolo_pt[1])**2)
            if dist < 50:
                w_x = (bg_pt[0]*bg_conf + yolo_pt[0]*yolo_conf) / (bg_conf + yolo_conf)
                w_y = (bg_pt[1]*bg_conf + yolo_pt[1]*yolo_conf) / (bg_conf + yolo_conf)
                final_pt = (int(w_x), int(w_y))
                final_conf = min(1.0, ((bg_conf + yolo_conf)/2))
                method = 'BG/YOLO'
                print(f'HYBRID: Corroborated detection at {final_pt}, confidence {final_conf:.2f}')
            else:
                # distance too large
                if bg_conf > yolo_conf:
                    final_pt = bg_pt
                    final_conf = bg_conf
                    method = 'BG'
                    print(f'BG: ball detected at {bg_pt}, confidence {final_conf:.2f}')
                else:
                    final_pt = yolo_pt
                    final_conf = yolo_conf
                    method = 'YOLO'
                    print(f'YOLO: ball detected at {final_pt}, confidence {final_conf:.2f}')
        elif bg_pt is not None:
            final_pt = bg_pt
            final_conf = bg_conf
            method = 'BG'
            print(f'BG: ball detected at {final_pt}, confidence {final_conf:.2f}')

        elif yolo_pt is not None:
            final_pt = yolo_pt
            final_conf = yolo_conf
            method = 'YOLO'
            print(f'YOLO: ball detected at {final_pt}, confidence {final_conf:.2f}')

        if final_pt is not None:
            self.successful(final_pt, method, frame, final_conf)
            return BallDetection(final_pt, timestamp, frame_no, self.frame_count, final_conf, method, self.camera_id)
        
        elif self.prev_grey is not None and self.lk_age < self.lk_max and self.lk_pts is not None:
            # use hsv segmentation + hough circle
            # or use LK optical flow
            self.lk_age += 1

            new_points, status, error = cv.calcOpticalFlowPyrLK(self.prev_grey, grey, self.lk_pts, None, **self.lk_params)
            print('No ball detected by BG or YOLO, using LK optical flow')
        
            x, y = new_points[0][0]

            if new_points is not None and status[0][0] == 1:

                centre = (int(x), int(y))
                print(f'Optical flow: ball detected at {centre}, age {self.lk_age}')
                h, w = grey.shape[:2]
                if not (10 < x < w-10 and 10 < y < h-10):
                    print("Point outside reasonable boundaries")
                    self.lk_pts = None
                    self.lk_age = 0
                    self.consec_no_detection += 1
                    self.max_consec_no_detection = max(self.max_consec_no_detection, self.consec_no_detection)
                    return None
                
                self.prev_grey = grey
                self.lk_pts = new_points

                estimated_conf = max(0.2, 0.5-(self.lk_age/10))
                if self.lk_age > self.lk_max:
                    estimated_conf = 0.1
            
                self.successful(centre, 'LK', frame, estimated_conf)
                return BallDetection(centre, timestamp, frame_no, self.frame_count, estimated_conf, 'LK', self.camera_id)
            else:
                self.lk_pts = None
                print('Optical flow: ball lost')
                # self.pts.appendleft((None, None))
                self.consec_no_detection += 1
                self.max_consec_no_detection = max(self.max_consec_no_detection, self.consec_no_detection)
                return None
        else:
            print('No ball detected')
            self.consec_no_detection += 1
            self.max_consec_no_detection = max(self.max_consec_no_detection, self.consec_no_detection)
            return None
        
    def run(self):
        detections_array = []
        print(f'Processing video {videos[self.camera_id]} from frame {frame_range[0]} to {frame_range[1]}')

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_range[0] - 1)
        self.frame_count = frame_range[0] - 1

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = self.cap.get(cv.CAP_PROP_POS_MSEC)
            frame_no = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))

            if frame_no > frame_range[1]:
                print(f'End of video reached at frame {frame_no}')
                break
            print(f'Processing frame {self.frame_count} @ {timestamp}ms')
            detection = self.process_frame(frame, timestamp, frame_no)
            
            # Create a clean display frame
            display_frame = frame.copy()
            
            # Draw the trails
            for i in range(1, len(self.pts)):
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue
                pt1, method1 = self.pts[i - 1]
                pt2, method2 = self.pts[i]

                # Calculate line thickness based on position in trail
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                # Set line colour based on detection method
                if method1 == 'BG':
                    line_colour = (255, 0, 0) # blue
                elif method1 == 'YOLO':
                    line_colour = (0, 0, 255) # red
                elif method1 == 'LK':
                    line_colour = (0, 255, 0) # green
                else:
                    line_colour = (255, 255, 0) # cyan

                # Draw connecting line
                cv.line(display_frame, pt1, pt2, line_colour, thickness)

            # Add detection rate text with larger font
            detection_rate = (self.successful_frames / (self.frame_count - frame_range[0] + 1)) * 100
            cv.putText(display_frame, f'Detection rate: {detection_rate:.2f}%', (20, 80), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
            
            # Add frame info
            cv.putText(display_frame, f'Frame: {self.frame_count}', (20, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
            
            # Mark current detection point
            if detection:
                cv.circle(display_frame, detection.centre, 15, (0, 0, 255), -1)
                detections_array.append(detection)
                # Add method and confidence info
                cv.putText(display_frame, f'{detection.detection_method}: {detection.confidence:.2f}', 
                          (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)

            # Save frame to video if enabled
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(display_frame)

            # Display frame
            cv.namedWindow('Hybrid tracking', cv.WINDOW_NORMAL) 
            cv.resizeWindow('Hybrid tracking', 2000, 700)
            cv.imshow('Hybrid tracking', display_frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                print(f'Quitting at frame {self.frame_count}')
                break

        metrics_path = os.path.join('output', f'detection_metrics_{self.camera_id}.csv')
        os.makedirs('output', exist_ok=True)

        detection_rate = (self.successful_frames / (self.frame_count - frame_range[0] + 1))*100

        avg_gap = sum(self.detection_gaps) / len(self.detection_gaps) if self.detection_gaps else 0
        print(f'Ball detection metrics for camera {self.camera_id}:')
        print(f'Frames processed: {self.frame_count}')
        print(f'Frames with detection: {self.successful_frames}')
        print(f'Successful detection rate: {detection_rate}%')
        print(f'Longest streak of no detection: {self.max_consec_no_detection} frames')
        print(f'Average detection gap length: {avg_gap:.2f} frames')  
         # Save metrics to CSV
        if args.save_output:
            with open(metrics_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Frames', self.frame_count - frame_range[0] + 1])
                writer.writerow(['Frames With Detection', self.successful_frames])
                writer.writerow(['Detection Rate (%)', f"{detection_rate:.2f}"])
                writer.writerow(['Max Consecutive Frames Without Detection', self.max_consec_no_detection])
                writer.writerow(['Average Gap Length', f"{avg_gap:.2f}"])
                # Add detailed gap information
                writer.writerow([])
                writer.writerow(['Gap Lengths (frames)'])
                for gap in self.detection_gaps:
                    writer.writerow([gap])
                    
            print(f"Detection metrics saved to: {metrics_path}")
        
        # Close video writer
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to: {self.video_filename}")
        
        self.cap.release()
        cv.destroyAllWindows()
        return detections_array


def main():
    base = 'output/matching_frames'
    counter = 1
    matching_dir = base
    while os.path.exists(matching_dir):
        counter += 1
        matching_dir = f'{base}{counter}'

    os.makedirs(matching_dir)
    print(f'Created output directory {matching_dir}')

    processor1 = CameraProcessor(1, args.video1, model_path=args.model, output_dir=matching_dir, save_video=args.save_video) # A1
    processor2 = CameraProcessor(2, args.video2, model_path=args.model, output_dir=matching_dir, save_video=args.save_video) # A2
    processor3 = CameraProcessor(3, args.video3, model_path=args.model, output_dir=matching_dir, save_video=args.save_video) # B1
    processor4 = CameraProcessor(4, args.video4, model_path=args.model, output_dir=matching_dir, save_video=args.save_video) # B2

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(processor1.run),
            executor.submit(processor2.run),
            executor.submit(processor3.run),
            executor.submit(processor4.run)
        ]
        results = [future.result() for future in futures]

    detectionsA = []
    detectionsB = []
    detectionsA.extend(results[0])
    detectionsA.extend(results[1])
    detectionsB.extend(results[2])
    detectionsB.extend(results[3])

    all_detections = []
    all_detections.extend(detectionsA)
    all_detections.extend(detectionsB)

    os.makedirs('output', exist_ok=True)
    
    print('Saving stereo A detections to CSV..')
    df_A = pd.DataFrame([
        {
            'x': d.centre[0],
            'y': d.centre[1],
            'timestamp': d.timestamp,
            'frame_no': d.frame_no,
            'frame_count': d.frame_count,
            'confidence': d.confidence,
            'detection_method': d.detection_method,
            'camera': d.camera
        } for d in detectionsA
    ])
    df_A.set_index('timestamp', inplace=True)
    df_A.sort_values(by='timestamp', inplace=True)
    df_stereoA = ss_csv(df_A, stereo='A')
    if args.save_output:
        df_stereoA.to_csv(output_A_csv, index=True)
        print(f'Saved pair A detections to {output_A_csv}')

    df_B = pd.DataFrame([
        {
            'x': d.centre[0],
            'y': d.centre[1],
            'timestamp': d.timestamp,
            'frame_no': d.frame_no,
            'frame_count': d.frame_count,
            'confidence': d.confidence,
            'detection_method': d.detection_method,
            'camera': d.camera
        } for d in detectionsB
    ])
    df_B.set_index('timestamp', inplace=True)
    df_B.sort_values(by='timestamp', inplace=True)
    df_stereoB = ss_csv(df_B, stereo='B')
    if args.save_output:
        df_stereoB.to_csv(output_B_csv, index=True)
        print(f'Saved pair B detections to {output_B_csv}')

if __name__ == '__main__':
    main()
    # save the matching stereo detections in new csv