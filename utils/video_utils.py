import cv2 as cv

def read_video(path):
    cap = cv.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()