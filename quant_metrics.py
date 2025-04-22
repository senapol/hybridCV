import cv2 as cv
import numpy as np
import argparse
import pandas as pd
import os
import time
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from adjustText import adjust_text

# Define detection methods as an enum for clarity
class DetectionMethod(Enum):
    YOLO = "YOLO"
    BG = "BG"
    YOLO_LK = "YOLO+LK"
    BG_LK = "BG+LK"
    HYBRID = "HYBRID"

@dataclass
class BallDetection:
    centre: tuple
    timestamp: float
    frame_no: int
    frame_count: int
    confidence: float
    detection_method: str
    camera: int

class DetectionMetrics:
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.frames_processed = 0
        self.successful_frames = 0
        self.detection_gaps = []
        self.consec_no_detection = 0
        self.max_consec_no_detection = 0
        self.total_processing_time = 0

    def add_detection(self):
        self.successful_frames += 1
        if self.consec_no_detection > 0:
            self.detection_gaps.append(self.consec_no_detection)
            self.consec_no_detection = 0

    def add_miss(self):
        self.consec_no_detection += 1
        self.max_consec_no_detection = max(self.max_consec_no_detection, self.consec_no_detection)

    def add_frame(self):
        self.frames_processed += 1

    def get_detection_rate(self):
        if self.frames_processed == 0:
            return 0
        return (self.successful_frames / self.frames_processed) * 100

    def get_avg_gap_length(self):
        if not self.detection_gaps:
            return 0
        return sum(self.detection_gaps) / len(self.detection_gaps)

    def get_summary_dict(self):
        return {
            'Method': self.method_name,
            'Frames Processed': self.frames_processed,
            'Successful Detections': self.successful_frames,
            'Detection Rate (%)': self.get_detection_rate(),
            'Max Consecutive Misses': self.max_consec_no_detection,
            'Avg Gap Length': self.get_avg_gap_length(),
            'Total Processing Time (s)': self.total_processing_time,
            'Avg Processing Time (ms/frame)': (self.total_processing_time * 1000 / self.frames_processed) if self.frames_processed > 0 else 0
        }

class ComparisonProcessor:
    def __init__(self, video_path: str, model_path: str = 'models/last3.pt', 
                 output_dir='output/method_comparison', visualize=True):
        self.output_dir = output_dir
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.visualize = visualize
        
        # Create metrics trackers for each method
        self.metrics = {
            DetectionMethod.YOLO: DetectionMetrics("YOLO Only"),
            DetectionMethod.BG: DetectionMetrics("Background Subtraction Only"),
            DetectionMethod.YOLO_LK: DetectionMetrics("YOLO with Lucas-Kanade"),
            DetectionMethod.BG_LK: DetectionMetrics("Background Subtraction with Lucas-Kanade"),
            DetectionMethod.HYBRID: DetectionMetrics("Hybrid Method (YOLO+BG+LK)")
        }
        
        # Detection history for each method
        self.history = {
            DetectionMethod.YOLO: deque(maxlen=64),
            DetectionMethod.BG: deque(maxlen=64),
            DetectionMethod.YOLO_LK: deque(maxlen=64),
            DetectionMethod.BG_LK: deque(maxlen=64),
            DetectionMethod.HYBRID: deque(maxlen=64)
        }
        
        # Lucas-Kanade state for each method that uses it
        self.lk_state = {
            DetectionMethod.YOLO_LK: {"age": 0, "pts": None, "prev_grey": None},
            DetectionMethod.BG_LK: {"age": 0, "pts": None, "prev_grey": None},
            DetectionMethod.HYBRID: {"age": 0, "pts": None, "prev_grey": None}
        }
        
        # Background subtraction
        self.backSub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)
        self.prev_frame = None
        self.prev_Mt = None
        
        # LK optical flow parameters
        self.lk_max = 10
        self.lk_params = dict(winSize=(21, 21),
                            maxLevel=3,
                            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))
        
        # Video properties
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Collector for all detections
        self.all_detections = {method: [] for method in DetectionMethod}
        
        print(f'Video initialized with WxH {self.width}x{self.height} @ {self.fps}fps')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.visualize:
            for method in DetectionMethod:
                os.makedirs(f'{self.output_dir}/{method.value}', exist_ok=True)
                
    def init_lk(self, frame, centre, method):
        """Initialize Lucas-Kanade tracking for a specific method"""
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.lk_state[method]["prev_grey"] = grey
        self.lk_state[method]["age"] = 0
        self.lk_state[method]["pts"] = np.array([[centre]], dtype=np.float32)
        
    def detect_yolo(self, frame):
        """YOLO detection only"""
        start_time = time.time()
        
        results = self.model(frame)[0]
        yolo_pt = None
        yolo_conf = 0.0
        
        if len(results.boxes) > 0 and results.boxes[0].conf[0].item() > 0.38:
            # Get highest confidence detection
            max_conf = max(results.boxes, key=lambda box: float(box.conf[0].item()))
            x1, y1, x2, y2 = max_conf.xyxy[0].cpu().numpy()
            yolo_pt = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            yolo_conf = max_conf.conf[0].item()
        
        process_time = time.time() - start_time
        self.metrics[DetectionMethod.YOLO].total_processing_time += process_time
        
        return yolo_pt, yolo_conf
        
    def detect_bgsub(self, frame, grey):
        """Background subtraction detection only"""
        start_time = time.time()
        
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
        confidence = 0.0
        
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
                M = cv.moments(best_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    ball_pt = (cx, cy)
                    confidence = best_circularity
        
        process_time = time.time() - start_time
        self.metrics[DetectionMethod.BG].total_processing_time += process_time
        
        return ball_pt, confidence
        
    def track_with_lk(self, frame, grey, method):
        """Apply Lucas-Kanade tracking for a method that has LK enabled"""
        start_time = time.time()
        
        # Skip if LK is not initialized for this method
        if (self.lk_state[method]["prev_grey"] is None or 
            self.lk_state[method]["pts"] is None or 
            self.lk_state[method]["age"] >= self.lk_max):
            return None, 0.0
            
        # Increment LK age
        self.lk_state[method]["age"] += 1
        
        # Calculate optical flow
        new_points, status, error = cv.calcOpticalFlowPyrLK(
            self.lk_state[method]["prev_grey"], 
            grey, 
            self.lk_state[method]["pts"], 
            None, 
            **self.lk_params
        )
        
        if new_points is not None and status[0][0] == 1:
            x, y = new_points[0][0]
            centre = (int(x), int(y))
            
            # Check if point is within frame boundaries
            h, w = grey.shape[:2]
            if not (10 < x < w-10 and 10 < y < h-10):
                # Reset LK tracking for this method
                self.lk_state[method]["pts"] = None
                self.lk_state[method]["age"] = 0
                process_time = time.time() - start_time
                
                # Update processing time based on the method
                if method == DetectionMethod.YOLO_LK:
                    self.metrics[DetectionMethod.YOLO_LK].total_processing_time += process_time
                elif method == DetectionMethod.BG_LK:
                    self.metrics[DetectionMethod.BG_LK].total_processing_time += process_time
                elif method == DetectionMethod.HYBRID:
                    self.metrics[DetectionMethod.HYBRID].total_processing_time += process_time
                    
                return None, 0.0
            
            # Update LK state for next frame
            self.lk_state[method]["prev_grey"] = grey
            self.lk_state[method]["pts"] = new_points
            
            # Estimate confidence based on age
            estimated_conf = max(0.2, 0.5-(self.lk_state[method]["age"]/10))
            if self.lk_state[method]["age"] > self.lk_max:
                estimated_conf = 0.1
                
            process_time = time.time() - start_time
            
            # Update processing time based on the method
            if method == DetectionMethod.YOLO_LK:
                self.metrics[DetectionMethod.YOLO_LK].total_processing_time += process_time
            elif method == DetectionMethod.BG_LK:
                self.metrics[DetectionMethod.BG_LK].total_processing_time += process_time
            elif method == DetectionMethod.HYBRID:
                self.metrics[DetectionMethod.HYBRID].total_processing_time += process_time
                
            return centre, estimated_conf
        else:
            # Reset LK tracking if point was lost
            self.lk_state[method]["pts"] = None
            
            process_time = time.time() - start_time
            
            # Update processing time based on the method
            if method == DetectionMethod.YOLO_LK:
                self.metrics[DetectionMethod.YOLO_LK].total_processing_time += process_time
            elif method == DetectionMethod.BG_LK:
                self.metrics[DetectionMethod.BG_LK].total_processing_time += process_time
            elif method == DetectionMethod.HYBRID:
                self.metrics[DetectionMethod.HYBRID].total_processing_time += process_time
                
            return None, 0.0
            
    def hybrid_detection(self, yolo_pt, yolo_conf, bg_pt, bg_conf):
        """Combined YOLO and background subtraction method"""
        start_time = time.time()
        
        final_pt = None
        final_conf = 0.0
        
        if bg_pt is not None and yolo_pt is not None:
            # If both methods detect something, compare their results
            dist = np.sqrt((bg_pt[0] - yolo_pt[0])**2 + (bg_pt[1] - yolo_pt[1])**2)
            
            if dist < 50:  # If detections are close, create weighted average
                w_x = (bg_pt[0]*bg_conf + yolo_pt[0]*yolo_conf) / (bg_conf + yolo_conf)
                w_y = (bg_pt[1]*bg_conf + yolo_pt[1]*yolo_conf) / (bg_conf + yolo_conf)
                final_pt = (int(w_x), int(w_y))
                final_conf = min(1.0, ((bg_conf + yolo_conf)/2))
            else:
                # If detections are far apart, choose the one with higher confidence
                if bg_conf > yolo_conf:
                    final_pt = bg_pt
                    final_conf = bg_conf
                else:
                    final_pt = yolo_pt
                    final_conf = yolo_conf
        elif bg_pt is not None:
            final_pt = bg_pt
            final_conf = bg_conf
        elif yolo_pt is not None:
            final_pt = yolo_pt
            final_conf = yolo_conf
            
        process_time = time.time() - start_time
        self.metrics[DetectionMethod.HYBRID].total_processing_time += process_time
        
        return final_pt, final_conf
        
    def save_visualization(self, frame, frame_count, method, pt, confidence):
        """Save visualization of detection for a specific method"""
        if not self.visualize:
            return
            
        frame_copy = frame.copy()
        
        # Draw detection point
        if pt is not None:
            cv.circle(frame_copy, pt, 15, (0, 0, 255), 5)
            cv.putText(frame_copy, f'{method.value}: {confidence:.2f}', (20, 40), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Draw trail for this method
        for i in range(1, len(self.history[method])):
            if self.history[method][i-1] is None or self.history[method][i] is None:
                continue
                
            pt1, _ = self.history[method][i-1]
            pt2, _ = self.history[method][i]
            
            if pt1 is None or pt2 is None:
                continue
                
            # Calculate thickness based on position in trail
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            
            # Set line color based on method
            if method == DetectionMethod.YOLO:
                color = (0, 0, 255)  # Red
            elif method == DetectionMethod.BG:
                color = (255, 0, 0)  # Blue
            elif method == DetectionMethod.YOLO_LK:
                color = (0, 0, 255)  # Red with green tint
            elif method == DetectionMethod.BG_LK:
                color = (255, 0, 255)  # Magenta
            else:  # Hybrid
                color = (255, 165, 0)  # Orange
                
            cv.line(frame_copy, pt1, pt2, color, thickness)
            
        # Save frame
        output_path = f'{self.output_dir}/{method.value}/frame_{frame_count}.jpg'
        cv.imwrite(output_path, frame_copy)
        
    def update_history(self, method, pt, frame, confidence=0.0):
        """Update detection history for a specific method"""
        if pt is not None:
            self.history[method].appendleft((pt, confidence))
            self.metrics[method].add_detection()
            
            # For methods with LK, initialize it when we have a detection
            if method in [DetectionMethod.YOLO_LK, DetectionMethod.BG_LK, DetectionMethod.HYBRID]:
                if isinstance(confidence, float):  # Only init if not already from LK
                    self.init_lk(frame, pt, method)
        else:
            self.history[method].appendleft((None, None))
            self.metrics[method].add_miss()
            
    def process_frame(self, frame, timestamp, frame_no, frame_count):
        """Process a single frame with all detection methods"""
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Update frame count for all metrics
        for method in DetectionMethod:
            self.metrics[method].add_frame()
            
        # 1. YOLO Only
        yolo_pt, yolo_conf = self.detect_yolo(frame)
        self.update_history(DetectionMethod.YOLO, yolo_pt, frame, yolo_conf)
        self.save_visualization(frame, frame_count, DetectionMethod.YOLO, yolo_pt, yolo_conf)
        
        if yolo_pt is not None:
            self.all_detections[DetectionMethod.YOLO].append(
                BallDetection(yolo_pt, timestamp, frame_no, frame_count, yolo_conf, "YOLO", 1)
            )
        
        # 2. Background Subtraction Only
        bg_pt, bg_conf = self.detect_bgsub(frame, grey)
        self.update_history(DetectionMethod.BG, bg_pt, frame, bg_conf)
        self.save_visualization(frame, frame_count, DetectionMethod.BG, bg_pt, bg_conf)
        
        if bg_pt is not None:
            self.all_detections[DetectionMethod.BG].append(
                BallDetection(bg_pt, timestamp, frame_no, frame_count, bg_conf, "BG", 1)
            )
        
        # 3. YOLO with Lucas-Kanade
        if yolo_pt is not None:
            # If YOLO detected something, use that
            yolo_lk_pt, yolo_lk_conf = yolo_pt, yolo_conf
        else:
            # Otherwise try to use LK optical flow
            yolo_lk_pt, yolo_lk_conf = self.track_with_lk(frame, grey, DetectionMethod.YOLO_LK)
            if yolo_lk_pt is not None:
                yolo_lk_conf = f"LK:{yolo_lk_conf:.2f}"
                
        self.update_history(DetectionMethod.YOLO_LK, yolo_lk_pt, frame, yolo_lk_conf)
        self.save_visualization(frame, frame_count, DetectionMethod.YOLO_LK, yolo_lk_pt, 
                                yolo_lk_conf if isinstance(yolo_lk_conf, float) else float(yolo_lk_conf.split(':')[1]))
        
        if yolo_lk_pt is not None:
            method_str = "YOLO" if isinstance(yolo_lk_conf, float) else "LK"
            conf_val = yolo_lk_conf if isinstance(yolo_lk_conf, float) else float(yolo_lk_conf.split(':')[1])
            self.all_detections[DetectionMethod.YOLO_LK].append(
                BallDetection(yolo_lk_pt, timestamp, frame_no, frame_count, conf_val, method_str, 1)
            )
        
        # 4. Background Subtraction with Lucas-Kanade
        if bg_pt is not None:
            # If BG detected something, use that
            bg_lk_pt, bg_lk_conf = bg_pt, bg_conf
        else:
            # Otherwise try to use LK optical flow
            bg_lk_pt, bg_lk_conf = self.track_with_lk(frame, grey, DetectionMethod.BG_LK)
            if bg_lk_pt is not None:
                bg_lk_conf = f"LK:{bg_lk_conf:.2f}"
                
        self.update_history(DetectionMethod.BG_LK, bg_lk_pt, frame, bg_lk_conf)
        self.save_visualization(frame, frame_count, DetectionMethod.BG_LK, bg_lk_pt, 
                               bg_lk_conf if isinstance(bg_lk_conf, float) else float(bg_lk_conf.split(':')[1]))
        
        if bg_lk_pt is not None:
            method_str = "BG" if isinstance(bg_lk_conf, float) else "LK"
            conf_val = bg_lk_conf if isinstance(bg_lk_conf, float) else float(bg_lk_conf.split(':')[1])
            self.all_detections[DetectionMethod.BG_LK].append(
                BallDetection(bg_lk_pt, timestamp, frame_no, frame_count, conf_val, method_str, 1)
            )
        
        # 5. Hybrid Method (YOLO + BG + LK)
        # First try to get a detection from the primary methods
        hybrid_pt, hybrid_conf = self.hybrid_detection(yolo_pt, yolo_conf, bg_pt, bg_conf)
        
        # If no primary detection, fall back to LK
        if hybrid_pt is None:
            hybrid_pt, hybrid_conf = self.track_with_lk(frame, grey, DetectionMethod.HYBRID)
            if hybrid_pt is not None:
                hybrid_conf = f"LK:{hybrid_conf:.2f}"
                
        self.update_history(DetectionMethod.HYBRID, hybrid_pt, frame, hybrid_conf)
        self.save_visualization(frame, frame_count, DetectionMethod.HYBRID, hybrid_pt, 
                               hybrid_conf if isinstance(hybrid_conf, float) else float(hybrid_conf.split(':')[1]))
        
        if hybrid_pt is not None:
            method_str = "HYBRID" if isinstance(hybrid_conf, float) else "LK"
            conf_val = hybrid_conf if isinstance(hybrid_conf, float) else float(hybrid_conf.split(':')[1])
            self.all_detections[DetectionMethod.HYBRID].append(
                BallDetection(hybrid_pt, timestamp, frame_no, frame_count, conf_val, method_str, 1)
            )
        
        # Display frame with detection rate overlays if visualize is enabled
        if self.visualize:
            display_frame = frame.copy()
            
            # Define colors for each method
            method_colors = {
                DetectionMethod.YOLO: (0, 0, 255),       # Red
                DetectionMethod.BG: (255, 0, 0),         # Blue
                DetectionMethod.YOLO_LK: (0, 255, 255),  # Yellow
                DetectionMethod.BG_LK: (255, 0, 255),    # Magenta
                DetectionMethod.HYBRID: (255, 165, 0)    # Orange
            }
            
            # Draw colored circles for each detection method
            if yolo_pt is not None:
                cv.circle(display_frame, yolo_pt, 5, method_colors[DetectionMethod.YOLO], 5)
                cv.putText(display_frame, "Y", (yolo_pt[0] - 5, yolo_pt[1] - 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, method_colors[DetectionMethod.YOLO], 2)
                
            if bg_pt is not None:
                cv.circle(display_frame, bg_pt, 10, method_colors[DetectionMethod.BG], 5)
                cv.putText(display_frame, "B", (bg_pt[0] - 5, bg_pt[1] - 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, method_colors[DetectionMethod.BG], 2)
                
            if yolo_lk_pt is not None:
                cv.circle(display_frame, yolo_lk_pt, 24, method_colors[DetectionMethod.YOLO_LK], 5)
                label = "YL" if isinstance(yolo_lk_conf, float) else "L"
                cv.putText(display_frame, label, (yolo_lk_pt[0] - 10, yolo_lk_pt[1] - 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, method_colors[DetectionMethod.YOLO_LK], 2)
                
            if bg_lk_pt is not None:
                cv.circle(display_frame, bg_lk_pt, 28, method_colors[DetectionMethod.BG_LK], 2)
                label = "BL" if isinstance(bg_lk_conf, float) else "L"
                cv.putText(display_frame, label, (bg_lk_pt[0] - 10, bg_lk_pt[1] - 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, method_colors[DetectionMethod.BG_LK], 2)
                
            if hybrid_pt is not None:
                cv.circle(display_frame, hybrid_pt, 32, method_colors[DetectionMethod.HYBRID], 3)
                label = "H" if isinstance(hybrid_conf, float) else "L"
                cv.putText(display_frame, label, (hybrid_pt[0] - 5, hybrid_pt[1] - 35), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, method_colors[DetectionMethod.HYBRID], 3)
            
            # Add detection rate text for each method
            y_pos = 30
            for method in DetectionMethod:
                rate = self.metrics[method].get_detection_rate()
                color = method_colors[method]
                
                # Draw colored circle next to method name in legend
                legend_circle_center = (20, y_pos - 5)
                cv.circle(display_frame, legend_circle_center, 8, color, -1)
                
                cv.putText(display_frame, f'{method.value}: {rate:.2f}%', (35, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 30
                
            # Add frame number
            cv.putText(display_frame, f'Frame: {frame_count}', (self.width - 200, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv.imshow('Method Comparison', display_frame)
            
        return {
            DetectionMethod.YOLO: yolo_pt is not None,
            DetectionMethod.BG: bg_pt is not None,
            DetectionMethod.YOLO_LK: yolo_lk_pt is not None,
            DetectionMethod.BG_LK: bg_lk_pt is not None,
            DetectionMethod.HYBRID: hybrid_pt is not None
        }
        
    def run(self, start_frame=0, end_frame=None, save_results=True):
        """Process video and collect metrics for all methods"""
        print(f'Processing video from frame {start_frame}' + 
              (f' to {end_frame}' if end_frame else ''))
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        
        while True:
            ret, frame = self.cap.read()
            if not ret or (end_frame and frame_count >= end_frame):
                break
                
            timestamp = self.cap.get(cv.CAP_PROP_POS_MSEC)
            frame_no = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            
            print(f'Processing frame {frame_count} @ {timestamp}ms')
            self.process_frame(frame, timestamp, frame_no, frame_count)
            
            frame_count += 1
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                print(f'Quitting at frame {frame_count}')
                break
                
        # Close video capture
        self.cap.release()
        cv.destroyAllWindows()
        
        if save_results:
            self.save_results()
            
        return self.metrics
        
    def save_results(self):
        """Save metrics and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create metrics DataFrame
        metrics_data = [metric.get_summary_dict() for metric in self.metrics.values()]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save metrics to CSV
        metrics_path = os.path.join(self.output_dir, f'comparison_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f'Metrics saved to {metrics_path}')
        
        # Save detection data for each method
        for method in DetectionMethod:
            if self.all_detections[method]:
                df = pd.DataFrame([
                    {
                        'x': d.centre[0],
                        'y': d.centre[1],
                        'timestamp': d.timestamp,
                        'frame_no': d.frame_no,
                        'frame_count': d.frame_count,
                        'confidence': d.confidence,
                        'detection_method': d.detection_method
                    } for d in self.all_detections[method]
                ])
                
                detections_path = os.path.join(self.output_dir, f'{method.value}_detections_{timestamp}.csv')
                df.to_csv(detections_path, index=False)
                print(f'{method.value} detections saved to {detections_path}')
                
        # Generate comparative visualization
        self.generate_comparison_plots(timestamp)
                
    def generate_comparison_plots(self, timestamp):
        """Generate comparative plots for the methods"""
        # Create figure for detection rates
        plt.figure(figsize=(12, 6))
        methods = [method.value for method in DetectionMethod]
        rates = [self.metrics[method].get_detection_rate() for method in DetectionMethod]
        
        # Plot detection rates
        plt.subplot(1, 2, 1)
        bars = plt.bar(methods, rates, color=['red', 'blue', 'yellow', 'magenta', 'orange'])
        plt.title('Detection Rate by Method (%)')
        plt.ylabel('Detection Rate (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot processing time
        plt.subplot(1, 2, 2)
        times = [self.metrics[method].total_processing_time * 1000 / self.metrics[method].frames_processed 
                for method in DetectionMethod]
        bars = plt.bar(methods, times, color=['red', 'blue', 'yellow', 'magenta', 'orange'])
        plt.title('Average Processing Time per Frame')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'detection_rates_{timestamp}.png'))
        
        # Create figure for gap metrics
        plt.figure(figsize=(12, 6))
        
        # Plot max consecutive misses
        plt.subplot(1, 2, 1)
        max_gaps = [self.metrics[method].max_consec_no_detection for method in DetectionMethod]
        bars = plt.bar(methods, max_gaps, color=['red', 'blue', 'yellow', 'magenta', 'orange'])
        plt.title('Max Consecutive Frames Without Detection')
        plt.ylabel('Frames')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        # Plot average gap length
        plt.subplot(1, 2, 2)
        avg_gaps = [self.metrics[method].get_avg_gap_length() for method in DetectionMethod]
        bars = plt.bar(methods, avg_gaps, color=['red', 'blue', 'yellow', 'magenta', 'orange'])
        plt.title('Average Gap Length')
        plt.ylabel('Frames')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'gap_metrics_{timestamp}.png'))

def main():
    parser = argparse.ArgumentParser(description='Compare different ball detection methods')
    parser.add_argument('--video', type=str, help='Path to video file', default='images/A1-s.mov')
    parser.add_argument('--model', type=str, help='Path to YOLO model file', default='models/last3.pt')
    parser.add_argument('--start', type=int, help='Starting frame number', default=1020)
    parser.add_argument('--end', type=int, help='Ending frame number', default=1680)
    parser.add_argument('--output_dir', type=str, help='Output directory', default='output/method_comparison')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    comparison = ComparisonProcessor(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
        visualize=not args.no_vis
    )
    
    metrics = comparison.run(start_frame=args.start, end_frame=args.end)
    
    # Print summary
    print("\n===== Detection Method Comparison Summary =====")
    print(f"Frames {args.start} to {args.end}")
    print("\nDetection Rates:")
    for method in DetectionMethod:
        rate = metrics[method].get_detection_rate()
        print(f"  {method.value}: {rate:.2f}%")
    
    print("\nMax Consecutive Frames Without Detection:")
    for method in DetectionMethod:
        max_gap = metrics[method].max_consec_no_detection
        print(f"  {method.value}: {max_gap}")
    
    print("\nAverage Gap Length:")
    for method in DetectionMethod:
        avg_gap = metrics[method].get_avg_gap_length()
        print(f"  {method.value}: {avg_gap:.2f} frames")
    
    print("\nAverage Processing Time Per Frame:")
    for method in DetectionMethod:
        time_ms = metrics[method].total_processing_time * 1000 / metrics[method].frames_processed
        print(f"  {method.value}: {time_ms:.2f} ms")
    
    print("\nComparison complete. Results saved to:", args.output_dir)

if __name__ == '__main__':
    main()