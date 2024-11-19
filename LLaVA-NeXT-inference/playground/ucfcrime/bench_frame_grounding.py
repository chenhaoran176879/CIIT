import os
import json
from video_utils import get_video_info

ucf_test_annot = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
uvcl_gt =  "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_mcq.jsonl"



def parse_timestamp_info(file_path):
    video_data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()  # 按空格分割
                video_file = parts[0]
                start_frame = int(parts[2])
                end_frame = int(parts[3])

                video_name = video_file
                video_data[video_name] = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
    return video_data


def merge_jsonl_with_txt(jsonl_file_path, txt_data):
    """将 TXT 数据合并到 JSONL 数据中"""
    merged_data = []
    output_filepath = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/gt_with_timestamp_fps.jsonl"
    with open(output_filepath, "a", encoding="utf-8") as f1:
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line.strip())
                video_name_key = json_data.get("video_name", "")
                if video_name_key in txt_data:
                    # 添加 start_frame 和 end_frame 信息
                    json_data.update(txt_data[video_name_key])
                    
                video_path = get_video_path(video_name_key)
                try:
                    video_info = get_video_info(video_path)
                    json_data.update(video_info)
                except Exception as e:
                    print(e)
                    
                merged_data.append(json_data)
                f1.write(json.dumps(json_data)+'\n')
                print(json_data)
             
            
    return merged_data

def save_merged_data(output_file_path, merged_data):
    """保存合并后的数据到新 JSONL 文件"""
    with open(output_file_path, "w", encoding="utf-8") as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_category(video_name):
        # 简化的分类获取逻辑，遇到第一个数字就截取
        for i, char in enumerate(video_name):
            if char.isdigit():
                return video_name[:i]
        return video_name  # 如果没有数字，返回完整名称
            
def get_video_path(video_name):
        # 简化的分类获取逻辑，遇到第一个数字就截取
        category = get_category(video_name)
        video_folder = '/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train'
        video_path = os.path.join(video_folder, category, video_name)
        if '.' not in video_path:
            video_path += '.mp4'
        return video_path

def add_timestamp_to_gt():
    video_data = parse_timestamp_info(ucf_test_annot)
    merged = merge_jsonl_with_txt(uvcl_gt,video_data)



add_timestamp_to_gt()