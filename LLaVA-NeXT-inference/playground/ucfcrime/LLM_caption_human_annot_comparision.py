import json
import pandas as pd

# 文件路径
human_annotation_path = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/human_annatation_train.jsonl"
llm_summarization_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results.jsonl"

# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

human_data = read_jsonl(human_annotation_path)
llm_data = read_jsonl(llm_summarization_path)

# 创建一个空列表用于存储对比结果
comparison_results = []

# 使用一个集合来跟踪处理过的 video_name
processed_video_names = set()

# 根据 video_name 进行匹配和比较
for human_entry in human_data:
    video_name = human_entry['video_name']
    
    # 如果 video_name 已经处理过，则跳过
    if video_name in processed_video_names:
        continue
    
    # 在 LLM 数据中找到匹配的 video_name
    llm_entry = next((item for item in llm_data if item['video_name'] in video_name), None)
    
    if llm_entry:
        # 提取 llm 的 response 数据，并清理冗余符号
        llm_response = json.loads(llm_entry['response'].replace('\\', ''))
        
        # 按顺序存储比较结果
        comparison_results.append({
            'video_name': video_name,
            '标注作者': '人工',
            'start_time': human_entry.get('start_time'),
            'end_time': human_entry.get('end_time'),
            'description': human_entry.get('description'),
            'confidence_score': human_entry.get('confidence_score'),
            'quality_score': human_entry.get('quality_score'),
            'is_repeated': human_entry.get('is_repeated'),
            'first_end_time': human_entry.get('first_end_time'),
            'timestamp': human_entry.get('timestamp'),
        })
        comparison_results.append({
            'video_name': video_name,
            '标注作者': 'LLM',
            'start_time': llm_response.get('start_time'),
            'end_time': llm_response.get('end_time'),
            'description': llm_response.get('description'),
        })
    
    # 将当前的 video_name 标记为已处理
    processed_video_names.add(video_name)

# 将比较结果转换为 DataFrame
df = pd.DataFrame(comparison_results)

# 按 video_name 排序
df = df.sort_values(by='video_name')

# 保存为 CSV 文件
output_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/video_caption_comparison.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"对比结果已成功存储为 CSV 文件：{output_path}")
