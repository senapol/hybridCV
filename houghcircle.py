import sys
import cv2 as cv
import numpy as np

def hough_circle(frame):
    # Convert to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 5)
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=30)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print('Circles detected: ', len(circles[0, :]))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3)
    else:
        print('No circles detected!')
    
    return frame

def main(argv):
    ## [load]
    print('Loading image...')
    default_file = 'images/tennis2.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    # gray = cv.medianBlur(gray, 5)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=50)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print('Circles detected: ', len(circles[0, :]))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    else:
        print('No circles detected!')
    ## [draw]

    ## [display]
    cv.imwrite('houghcircle2.png', src)
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    ## [display]

    return 0


if __name__ == "__main__":
    # main(sys.argv[1:])
    vs = cv.VideoCapture('images/tennisvid7.mp4')
    while True:
        ret, frame = vs.read()
        if frame is None:
            break
        frame = hough_circle(frame)
        cv.imshow('Frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break