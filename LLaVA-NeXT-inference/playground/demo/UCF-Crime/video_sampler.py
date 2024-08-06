import cv2
from PIL import Image



def is_valid_video_filename(name):
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    ext = name.split('.')[-1].lower()
    
    if ext in video_extensions:
        return True
    else:
        return False

def sample_frames_from_video(video_file, num_frames) :
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames


def sample_frames_from_video_with_splits(video_file, num_frames,num_splits):
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    total_num_frames = num_splits * num_frames
    

