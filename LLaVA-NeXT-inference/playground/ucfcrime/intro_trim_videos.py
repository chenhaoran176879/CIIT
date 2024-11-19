from video_utils import trim_and_save_video
import os

sample_videos_with_timestamp = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/sample_videos.txt"

def trim_videos(video_info_file,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(video_info_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            video_name, category, start_frame, end_frame = parts[0], parts[1], int(parts[2]), int(parts[3])
            video_path = f"/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/{category}/{video_name}"
            output_video_path = os.path.join(output_dir, f"trimmed_{video_name}")

            # 调用 trim_and_save_video 函数
            trim_and_save_video(video_path, output_video_path, start_frame=start_frame, end_frame=end_frame)
            print(f"SAVED {output_video_path}")

save_dir = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/"
trim_videos(sample_videos_with_timestamp,save_dir)
