from ultralytics import YOLO
from utils import read_video, save_video
from trackers import PlayerTracker

# model = YOLO('models/best.pt') #yolov5

# result = model.predict('images/tennisvid3.mp4', conf=0.2, save=True)

# print(result)

# print('boxes: ')

# for box in result[0].boxes:
#     print(box)

def main():
    input_path = 'images/tennisvid.mp4'

    # Read video
    frames = read_video(input_path)

    player_tracker = PlayerTracker('yolov8x.pt')

    player_detections = player_tracker.detect_frames(frames)

    output_frames = player_tracker.draw_bboxes(frames, player_detections)

    save_video(frames, 'output_videos/output.avi')

if __name__ == '__main__':
    main() 