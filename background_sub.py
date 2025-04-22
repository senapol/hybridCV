import cv2
import numpy as np
import os
from datetime import datetime
import csv

def tennis_ball_tracking(video_path, save_output=True, output_path=None, save_metrics=True):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)
    
    prev_frame = None
    prev_Mt = None
    ball_positions = []
    
    total_frames = 0
    frames_with_detection = 0
    consecutive_no_detection = 0
    max_consecutive_no_detection = 0
    detection_gaps = []
    
    # video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = None
    if save_output:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tennis_ball_tracking_{timestamp}.mp4"
            output_path = os.path.join(os.path.dirname(video_path), output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        est_width = min(frame_width * 3, 1600)
        ratio = est_width / (frame_width * 3)
        est_height = int(frame_height * 2 * ratio)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (est_width, est_height))
        
        print(f"Tennis ball tracking started. Saving output to {output_path}")
    else:
        print("Tennis ball tracking started. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or error reading frame.")
            break
            
        total_frames += 1
        
        original = frame.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Background subtraction with threshold
        fgMask = backSub.apply(frame)
        _, Mb = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        
        # Step 2: Dilation and erosion to connect pixels around players
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        dilated = cv2.dilate(Mb, kernel_dilate, iterations=2)
        eroded = cv2.erode(dilated, kernel_erode, iterations=1)
        
        # Step 3: Big object removal to remove players from Mb to get M2b
        # Find contours and filter out large objects (players)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M2b = np.zeros_like(eroded)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter based on area - keep small objects (potential balls)
            # Adjust these thresholds based on your video resolution and ball size
            if 10 < area < 500:  # Adjust these values as needed
                cv2.drawContours(M2b, [contour], -1, 255, -1)
        
        # Step 4: Difference between consecutive frames to get moving objects (Md)
        Md = np.zeros_like(gray)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(gray, prev_frame)
            _, frame_diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            Md = frame_diff_thresh
        
        prev_frame = gray.copy()
        
        # Step 5: Calculate refined map Mt = AND(M2b, Md)
        Mt = cv2.bitwise_and(M2b, Md)
        
        # Step 6: Build Mpre to clean up pixels from previous frames
        Mpre = np.ones_like(Mt) * 255
        if prev_Mt is not None:
            # Dilate previous Mt to get areas where ball cannot be
            prev_Mt_dilated = cv2.dilate(prev_Mt, kernel_dilate, iterations=1)
            # Invert to get where ball can be
            Mpre = cv2.bitwise_not(prev_Mt_dilated)
        
        # Step 7: Final map Mf = AND(Mt, Mpre)
        Mf = cv2.bitwise_and(Mt, Mpre)
        
        # Update prev_Mt for next iteration
        prev_Mt = Mt.copy()
        
        # Find potential ball in Mf
        ball_contours, _ = cv2.findContours(Mf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # visualization images
        Mb_vis = cv2.cvtColor(Mb, cv2.COLOR_GRAY2BGR)
        M2b_vis = cv2.cvtColor(M2b, cv2.COLOR_GRAY2BGR)
        Md_vis = cv2.cvtColor(Md, cv2.COLOR_GRAY2BGR)
        Mt_vis = cv2.cvtColor(Mt, cv2.COLOR_GRAY2BGR)
        Mf_vis = cv2.cvtColor(Mf, cv2.COLOR_GRAY2BGR)
        
        # Track the ball - get the largest contour from Mf
        ball_pos = None
        if ball_contours:
            best_circularity = 0
            best_contour = None
            
            for contour in ball_contours:
                area = cv2.contourArea(contour)
                if area < 5:  # Filter very small noise
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_contour = contour
            
            if best_contour is not None and best_circularity > 0.5:  # Ball should be somewhat circular
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    ball_pos = (cx, cy)
                    ball_positions.append(ball_pos)
                    
                    # Successfully detected ball
                    frames_with_detection += 1
                    
                    if consecutive_no_detection > 0:
                        detection_gaps.append(consecutive_no_detection)
                        consecutive_no_detection = 0
                    
                    cv2.circle(original, ball_pos, 5, (0, 0, 255), -1)
        
        # Draw ball trajectory
        if len(ball_positions) >= 2:
            for i in range(1, min(len(ball_positions), 20)):  # Show last 20 positions
                cv2.line(original, ball_positions[-i], ball_positions[-i-1], (0, 255, 0), 2)
        
        if ball_pos is None:
            consecutive_no_detection += 1
            max_consecutive_no_detection = max(max_consecutive_no_detection, consecutive_no_detection)
            
        detection_rate = (frames_with_detection / total_frames) * 100 if total_frames > 0 else 0
        
        cv2.putText(original, f"Detection Rate: {detection_rate:.1f}%", (10, original.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # First row: Original, Background Subtraction (Mb), Big Object Removal (M2b)
        first_row = np.hstack([original, Mb_vis, M2b_vis])
        
        # Second row: Frame Difference (Md), Refined Map (Mt), Final Map (Mf)
        second_row = np.hstack([Md_vis, Mt_vis, Mf_vis])
        
        # Combine rows
        combined = np.vstack([first_row, second_row])
        
        # Resize if too large
        if combined.shape[1] > 1600:
            scale = 1600 / combined.shape[1]
            combined = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        
        h_unit = combined.shape[0] // 2
        w_unit = combined.shape[1] // 3
        
        # First row labels
        cv2.putText(combined, "Original + Ball Tracking", (10, 30), font, font_scale, font_color, 1)
        cv2.putText(combined, "Mb: Background Subtraction", (w_unit + 10, 30), font, font_scale, font_color, 1)
        cv2.putText(combined, "M2b: After Big Object Removal", (2*w_unit + 10, 30), font, font_scale, font_color, 1)
        
        # Second row labels
        cv2.putText(combined, "Md: Frame Difference", (10, h_unit + 30), font, font_scale, font_color, 1)
        cv2.putText(combined, "Mt: Refined Map", (w_unit + 10, h_unit + 30), font, font_scale, font_color, 1)
        cv2.putText(combined, "Mf: Final Ball Map", (2*w_unit + 10, h_unit + 30), font, font_scale, font_color, 1)
        
        cv2.imshow('Tennis Ball Tracking', combined)
        
        if save_output and video_writer is not None:
            video_writer.write(combined)
        
        # Exit on 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save_output and video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {output_path}")
    cv2.destroyAllWindows()
    
    # Calculate final metrics
    detection_rate = (frames_with_detection / total_frames) * 100 if total_frames > 0 else 0
    avg_gap_length = sum(detection_gaps) / len(detection_gaps) if detection_gaps else 0
    
    print("\nBall Detection Metrics:")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with successful ball detection: {frames_with_detection}")
    print(f"Overall detection rate: {detection_rate:.2f}%")
    print(f"Longest streak without detection: {max_consecutive_no_detection} frames")
    print(f"Average gap length between detections: {avg_gap_length:.2f} frames")
    
    if save_metrics:
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv" if output_path else os.path.join(
            os.path.dirname(video_path), 
            f"ball_detection_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Frames', total_frames])
            writer.writerow(['Frames With Detection', frames_with_detection])
            writer.writerow(['Detection Rate (%)', f"{detection_rate:.2f}"])
            writer.writerow(['Max Consecutive Frames Without Detection', max_consecutive_no_detection])
            writer.writerow(['Average Gap Length', f"{avg_gap_length:.2f}"])
            writer.writerow([])
            writer.writerow(['Gap Lengths (frames)'])
            for gap in detection_gaps:
                writer.writerow([gap])
                
        print(f"Detection metrics saved to: {metrics_path}")
    
    return {
        'total_frames': total_frames,
        'frames_with_detection': frames_with_detection,
        'detection_rate': detection_rate,
        'max_consecutive_no_detection': max_consecutive_no_detection,
        'avg_gap_length': avg_gap_length,
        'detection_gaps': detection_gaps
    }

if __name__ == "__main__":
    video_path = "images/shortcrop.mov"
    
    save_output = True
    
    output_path = None
    
    save_metrics = True
    
    metrics = tennis_ball_tracking(video_path, save_output, output_path, save_metrics)
    
    print(f"\nDetection rate: {metrics['detection_rate']:.2f}%")