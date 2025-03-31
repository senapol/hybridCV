import cv2 as cv
from dataclasses import dataclass
from ultralytics import YOLO
import time
from typing import Optional
from collections import deque
import numpy as np
import argparse as ap
import pandas as pd

@dataclass
class BallDetection:
    centre: tuple
    timestamp: float
    frame: int
    confidence: float
    detection_method: str # YOLO, HSV, LK, Hough
    camera: int

# Actual frame of image or pointer to frame capture.get(CV_CAP_PROP_POS_FRAMES);

class CameraProcessor:
    def __init__(self, camera_id: int, video_path: str, model_path: str = 'models/last2.pt'):
        self.camera_id = camera_id
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.frame_count = 0
        self.successful_frames = 0
        self.start_time = time.time()
        self.pts = deque(maxlen=64)

        # lk optical flow
        self.prev_grey = None
        self.lk_age = 0
        self.lk_max = 30
        self.lk_pts = None
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))

        #properties
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        print(f'Camera {camera_id} initialized with WxH {self.width}x{self.height} @ {self.fps}fps')

    def init_lk(self, frame, centre):
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.prev_grey = grey
        self.lk_age = 0
        self.lk_pts = np.array([[centre]], dtype=np.float32)
        print(f'Initialising LK at {centre}')


    def process_frame(self, frame, timestamp, cam_id) -> Optional[BallDetection]:

        self.frame_count += 1
        # current = time.time() - self.start_time

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        results = self.model(frame)[0]

        if len(results.boxes) > 0 and results.boxes[0].conf[0].item() > 0.35:
            # get highest confidence detection
            max_conf = max(results.boxes, key=lambda box: float(box.conf[0].item()))
            x1, y1, x2, y2 = max_conf.xyxy[0].cpu().numpy()
            centre = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            conf = max_conf.conf[0].item()

            print(f'YOLO: ball detected at {centre} with confidence {conf}')
            self.pts.appendleft((centre, 'YOLO'))

            self.init_lk(frame, centre)
            self.successful_frames += 1
            return BallDetection(centre, timestamp, conf, 'YOLO', cam_id)
        
        elif self.prev_grey is not None and self.lk_age < self.lk_max and self.lk_pts is not None:
            # use hsv segmentation + hough circle
            # or use LK optical flow
            self.lk_age += 1

            new_points, status, error = cv.calcOpticalFlowPyrLK(self.prev_grey, grey, self.lk_pts, None, **self.lk_params)
            print('No ball detected by YOLO, using LK optical flow')
        
            x, y = new_points[0][0]

            if new_points is not None and status[0][0] == 1:
                # if self.pts and self.pts[0][0] is not None:
                #     prev_x, prev_y = self.pts[0][0]
                #     move_dist = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                #     if move_dist > 100:
                #         print(f'Optical flow: ball moved too far: {move_dist} pixels, ball lost')
                #         self.lk_pts = None
                #         return None

                centre = (int(x), int(y))
                print(f'Optical flow: ball detected at {centre}, age {self.lk_age}')
                h, w = grey.shape[:2]
                if not (10 < x < w-10 and 10 < y < h-10):
                    print("Point outside reasonable boundaries")
                    self.lk_pts = None
                    # self.pts.appendleft((None, None))
                    return None
                self.pts.appendleft((centre, 'LK'))

                self.prev_grey = grey
                self.lk_pts = new_points

                estimated_conf = max(0.2, 0.5-(self.lk_age/10))
                if self.lk_age > self.lk_max:
                    estimated_conf = 0.1
                
                self.successful_frames += 1
                return BallDetection(centre, timestamp, estimated_conf, 'LK', cam_id)
            else:
                self.lk_pts = None
                print('Optical flow: ball lost')
                # self.pts.appendleft((None, None))
                return None
        else:
            print('No ball detected')
            # self.pts.appendleft((None, None))
            return None
        
    def run(self):
        detections_array = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = self.cap.get(cv.CAP_PROP_POS_MSEC)
            print(f'Processing frame {self.frame_count} @ {timestamp}ms')

            detection = self.process_frame(frame, timestamp, self.camera_id)

            
            for i in range(1, len(self.pts)):
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue
                    
                    pt1, method1 = self.pts[i - 1]
                    pt2, method2 = self.pts[i]

                    # if len(pt1) != 2 or len(pt2) != 2:
                    #     print(f'Point length error: {len(pt1)}, {len(pt2)}')
                    #     continue

                    # Calculate line thickness based on position in trail
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    
                    # Set line colour based on detection method
                    if method1 == 'YOLO':
                        line_colour = (0, 0, 255) # red
                    
                    elif method2 == 'LK':
                        line_colour = (0, 255, 0) # green
                    
                    else:
                        line_colour = (255, 0, 0) # blue

                    # Draw connecting line
                    cv.line(frame, pt1, pt2, line_colour, thickness)

            if detection:
                cv.circle(frame, detection.centre, 5, (0, 0, 255), -1)
                detections_array.append(detection)
                # cv.putText(frame, f'Method|Confidence: {detection.detection_method}|{detection.confidence:.2f}', (detection.centre[0], detection.centre[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv.namedWindow('Hybrid tracking', cv.WINDOW_NORMAL) 
  
            cv.resizeWindow('Hybrid tracking', 2000, 700)
            # cv.putText(frame, f'FRame: {self.frame_count} @ {timestamp:.2f}s', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.imshow('Hybrid tracking', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print(f'Quitting at frame {self.frame_count}')
                break

        print(f'Successful detection rate: {(self.successful_frames/self.frame_count)*100}%')        
        self.cap.release()
        cv.destroyAllWindows()
        return detections_array


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file', default='images/tennisvid7.mp4')
    args = parser.parse_args()

    processor0 = CameraProcessor(0, args.video)
    # processor1 = CameraProcessor(1, 'images/tennisvid.mp4')
    # processor2 = CameraProcessor(2, 'images/tennisvid.mp4')
    # processor3 = CameraProcessor(3, 'images/tennisvid.mp4')

    detections0 = processor0.run()
    print('Saving to csv..')
    detections_df = pd.DataFrame([
        {
            'x': d.centre[0],
            'y': d.centre[1],
            'timestamp': d.timestamp,
            'confidence': d.confidence,
            'detection_method': d.detection_method,
            'camera': d.camera
        } for d in detections0
    ])
    detections_df.to_csv(f'output/detections.csv', index=False)

    # detections1 = processor1.run()
    # detections2 = processor2.run()
    # detections3 = processor3.run()

    # final_array = (detections0, detections1, detections2, detections3)
    # final_array.sort(key=lambda x: x.timestamp)


if __name__ == '__main__':
    main()