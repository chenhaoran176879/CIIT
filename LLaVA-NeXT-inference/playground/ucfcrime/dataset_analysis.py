import json
from collections import defaultdict
import re


def count_category(gt_path):

    # 统计结果字典，形式为 {data_split: {video_category: count}}
    video_count_by_split = defaultdict(lambda: defaultdict(int))

    # 读取 jsonl 文件
    with open(gt_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            
            # 获取data_split和video_name
            data_split = data.get('data_split', 'unknown')
            video_name = data.get('video_name', '')
            
            # 提取视频类别（第一个数字之前的名称）
            match = re.match(r'([a-zA-Z]+)', video_name)
            if match:
                video_category = match.group(1)
                
                # 更新统计结果
                video_count_by_split[data_split][video_category] += 1

    # 打印结果
    for split, categories in video_count_by_split.items():
        print(f"Data split: {split}")
        for category, count in categories.items():
            print(f"  {category}: {count}")

count_category("/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_mcq.jsonl")