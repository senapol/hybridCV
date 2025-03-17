from ultralytics import YOLO
import cv2 as cv
import numpy as np
from collections import deque
import pandas as pd

def YOLOdetection(path):
# YOLOv6n is fastest but least accurate model
# mAP accuracy can be increased by training
    model = YOLO('yolov8x.pt') #yolov5

    # if __name__ == '__main__':
    #     results = model.train(data="config.yaml", epochs=50, patience=5)

    results = model.predict(path, conf=0.5, save=True)

# def hsvHoughCircle(path):




def track_ball_motion(video_path, model_path='models/last2.pt', buffer_size=64, conf_thresh=0.4):
    model = YOLO(model_path)
    cap = cv.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}')
    frames_with_detection = 0

    # Initialize deque to store ball centers
    pts = deque(maxlen=buffer_size)

    prev_centre = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame)[0]
        
        if len(results.boxes) > 0:

            frames_with_detection += 1

            # Get highest confidence detection
            valid_boxes = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > conf_thresh:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    valid_boxes.append({
                        'box': (x1, y1, x2, y2),
                        'centre': centre,
                        'conf': conf
                    })

            if valid_boxes:
                if prev_centre is not None:
                    best_box = min(valid_boxes, 
                                   key=lambda x: (np.linalg.norm(np.array(x['centre']) - np.array(prev_centre))))
                else:
                    best_box = max(valid_boxes, key=lambda x: x['conf'])

                x1, y1, x2, y2 = best_box['box']
                centre = best_box['centre']

                prev_centre = centre
                pts.appendleft(centre)
            
            # Draw bounding box
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw center point
                cv.circle(frame, centre, 4, (0, 0, 255), -1)
                
                # Draw detection confidence
                cv.putText(frame, f'{best_box["conf"]:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw motion trail
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                        
                    # Calculate line thickness based on position in trail
                    thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
                    
                    # Draw connecting line
                    cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
                for box in valid_boxes:
                    if box != best_box:
                        x1, y1, x2, y2 = box['box']
                        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # else:
        #     box = hsvHoughCircle(frame)
        #     if box is not None:
        #         x1, y1, x2, y2 = box
        #         centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        #         prev_centre = centre
        #         pts.appendleft(centre)
        #         cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #         cv.circle(frame, centre, 4, (0, 0, 255), -1)
        #         for i in range(1, len(pts)):
        #             if pts[i - 1] is None or pts[i] is None:
        #                 continue
        #             thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        #             cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        detection_percentage = (frames_with_detection / total_frames) * 100
        print(f'Frames with detection: {frames_with_detection}\nDetection: {detection_percentage:.2f}%') 
        cv.imshow('Ball Motion', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    track_ball_motion('images/tennisvid7.mp4')

# YOLOdetection('images/tennis2.png')
# hsv_image = cv.cvtColor(cv.imread('images/tennis2.png'), cv.COLOR_BGR2HSV)
# cv.imwrite('images/hsv.png', hsv_image)

class LKtracker:

    def __init__(self):
        # params for ShiTomasi corner detection
        self.feat_params = dict(
            maxCorners=1,
            qualityLevel=0.1,
            minDistance=15,
            blockSize=3
        )

        # params for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        self.prev_gray = None
        self.prev_points = None

    def detect_init_points(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

        p0 = cv.goodFeaturesToTrack(gray, mask=None, **self.feat_params)
        self.prev_gray = gray
        self.prev_points = p0
        return p0
    
    def track(self, frame):
        if self.prev_gray is None:
            return self.detect_init_points(frame)
        
        if self.prev_points is None or len(self.prev_points) == 0:
            points = self.detect_init_points(frame)
            return points
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        # select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.prev_points[st == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv.line(mask, (a, b), (c, d), (0,0,255), 2)
                frame = cv.circle(frame, (a, b), 5, (0,0,255), -1)
            img = cv.add(frame, mask)

            # cv.imshow('frame', img)

            self.prev_gray = gray
            self.prev_points = good_new.reshape(-1, 1, 2)

            return img, good_new
        
        return frame, None
    
if __name__ == '__main__':
    cap = cv.VideoCapture('images/ball.mp4')
    tracker = LKtracker()

    while True:
        ret, frame = cap.read()

        if not ret:
            print('No frames grabbed!')
            break

        points = tracker.track(frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()

class HybridTracker:
    def __init__(self, confidence_threshold=0.5, max_frames_without_detection=10):
        # Initialize YOLO model
        # self.model = YOLO('models/best.pt')
        # self.conf_threshold = confidence_threshold
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),  # Smaller window for faster moving objects
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Tracking state
        self.prev_gray = None
        self.prev_points = None
    #     self.frames_without_detection = 0
    #     self.max_frames_without_detection = max_frames_without_detection
    #     self.tracking_lost = False

    # def detect_ball(self, frame):
    #     """Run YOLO detection on frame"""
    #     results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
    #     if len(results.boxes.data) > 0:
    #         # Get the box with highest confidence
    #         box = results.boxes.data[0]  # [x1, y1, x2, y2, confidence, class_id]
    #         return [int(x) for x in box[:4]]  # Return [x1, y1, x2, y2]
    #     return None

    def init_from_yolo(self, frame, yolo_box):
        """Initialize tracking from YOLO detection"""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Get center of yolo box
        centre_x = int((yolo_box[0] + yolo_box[2]) / 2)
        centre_y = int((yolo_box[1] + yolo_box[3]) / 2)
        
        h, w = frame.shape[:2]
        if not (0 < centre_x < w and 0 < centre_y < h):
            print('Initial point outside of frame')
            return None, None
        
        self.prev_points = np.array([[[centre_x, centre_y]]], dtype=np.float32)
        self.prev_gray = gray
        self.frames_without_detection = 0
        self.tracking_lost = False
        
        # Draw detection
        mask = np.zeros_like(frame)
        frame = cv.circle(frame, (centre_x, centre_y), 3, (0, 255, 0), -1)
        return frame, self.prev_points

    def track(self, frame):
        """Main tracking method combining YOLO and Lucas-Kanade"""
        output_frame = frame.copy()
        
        # Run YOLO detection periodically or if tracking is lost
        if self.tracking_lost or self.frames_without_detection >= self.max_frames_without_detection:
            yolo_box = self.detect_ball(frame)
            if yolo_box is not None:
                return self.init_from_yolo(output_frame, yolo_box)
            elif self.prev_points is None:
                return frame, None
        
        if self.prev_gray is None or self.prev_points is None:
            return frame, None
        
        # Prepare frame for optical flow
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_points, status, error = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if new_points is not None and status.all():
            # Draw tracking
            for new, old in zip(new_points, self.prev_points):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                
                # Check if point moved too far (likely false tracking)
                distance = np.sqrt((a - c)**2 + (b - d)**2)
                if distance > 100:  # Threshold for maximum movement
                    self.tracking_lost = True
                    return output_frame, None
                
                output_frame = cv.circle(output_frame, (a, b), 3, (0, 255, 0), -1)
                output_frame = cv.line(output_frame, (a, b), (c, d), (0, 255, 0), 2)
            
            self.prev_gray = gray
            self.prev_points = new_points
            self.frames_without_detection += 1
            return output_frame, new_points
        
        # Tracking failed
        self.tracking_lost = True
        return output_frame, None

# def main():
#     cap = cv.VideoCapture('images/ball.mp4')
#     if not cap.isOpened():
#         print('Error opening video file')
#         return
    
#     # Get video properties
#     width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv.CAP_PROP_FPS)
    
#     # Initialize tracker
#     tracker = HybridTracker(confidence_threshold=0.5, max_frames_without_detection=5)
    
#     # Initialize video writer
#     out = cv.VideoWriter(
#         'output.mp4',
#         cv.VideoWriter_fourcc(*'mp4v'),
#         fps,
#         (width, height)
#     )
    
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f'Processed {frame_count} frames')
#             break
            
#         # Process frame
#         output_frame, points = tracker.track(frame)
        
#         if output_frame is not None:
#             # Add frame counter
#             cv.putText(
#                 output_frame,
#                 f'Frame: {frame_count}',
#                 (10, 30),
#                 cv.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2
#             )
#             out.write(output_frame)
            
#         frame_count += 1
        
#     cap.release()
#     out.release()

# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs=50, patience=5)