from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def callback(value):
    pass

def setup_trackbars():
    cv2.namedWindow("Tennis Ball Detector", 0)
    
    # Create trackbars for HSV ranges
    # Starting values for yellow tennis ball detection
    cv2.createTrackbar("H_MIN", "Tennis Ball Detector", 20, 179, callback)  # Hue range is 0-179 in OpenCV
    cv2.createTrackbar("H_MAX", "Tennis Ball Detector", 40, 179, callback)
    cv2.createTrackbar("S_MIN", "Tennis Ball Detector", 100, 255, callback)
    cv2.createTrackbar("S_MAX", "Tennis Ball Detector", 255, 255, callback)
    cv2.createTrackbar("V_MIN", "Tennis Ball Detector", 100, 255, callback)
    cv2.createTrackbar("V_MAX", "Tennis Ball Detector", 255, 255, callback)

def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H_MIN", "Tennis Ball Detector")
    h_max = cv2.getTrackbarPos("H_MAX", "Tennis Ball Detector")
    s_min = cv2.getTrackbarPos("S_MIN", "Tennis Ball Detector")
    s_max = cv2.getTrackbarPos("S_MAX", "Tennis Ball Detector")
    v_min = cv2.getTrackbarPos("V_MIN", "Tennis Ball Detector")
    v_max = cv2.getTrackbarPos("V_MAX", "Tennis Ball Detector")
    return h_min, s_min, v_min, h_max, s_max, v_max

def get_hsv_range(args):
    # Parse command line arguments
    
    # Set up video capture or load image
    if args['webcam']:
        cap = cv2.VideoCapture(0)
    elif args['image']:
        image = cv2.imread(args['image'])
    else:
        print("Please specify either --image or --webcam")
        return

    setup_trackbars()
    
    while True:
        if args['webcam']:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = image.copy()
            
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current trackbar values
        h_min, s_min, v_min, h_max, s_max, v_max = get_trackbar_values()
        
        # Create mask
        lower = (h_min, s_min, v_min)
        upper = (h_max, s_max, v_max)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply mask to original image
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw largest contour (assumed to be the tennis ball)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            if len(contours) == 1 or cv2.contourArea(largest_contour) > 3:
                # print(f"Largest contour area: {cv2.contourArea(largest_contour)}")
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            
        
        # Show results
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)
        
        # Print current values when 'p' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print(f"HSV Range: ({h_min}, {s_min}, {v_min}) - ({h_max}, {s_max}, {v_max})")
        elif key == ord('q'):
            break
    
    if args['webcam']:
        cap.release()
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help='Path to the image')
    ap.add_argument('-w', '--webcam', action='store_true', help='Use webcam')
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())

    if args['image']:
        get_hsv_range(args)
        return None
    # get_hsv_range(args)

    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points
    yellowLower = (19,31,153)
    yellowUpper = (45,255,255)
    pts = deque(maxlen=args["buffer"])
    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        vs = VideoStream(src=0).start()
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["video"])
    # allow the camera or video file to warm up
    time.sleep(2.0)

    while True:
	# grab the current frame
        print("Grabbing frame")
        frame = vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
        # resize the frame, blur it, and convert it to the HSV
        # color space
        print("Resizing frame")
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            # if radius > 0:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # update the points queue
        pts.appendleft(center)       
	# loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # show the frame to our screen
        print("Showing frame")
        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    # if we are not using a video file, stop the camera video stream
    if not args.get("video", False):
        vs.stop()
    # otherwise, release the camera
    else:
        vs.release()
    # close all windows
    cv2.destroyAllWindows()

def notmain():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help='Path to the image')
    ap.add_argument('-w', '--webcam', action='store_true', help='Use webcam')
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())
    
    get_hsv_range(args)
    
if __name__ == '__main__':
    main()