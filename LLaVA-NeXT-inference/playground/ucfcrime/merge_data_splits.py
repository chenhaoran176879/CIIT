import json

# 文件路径
file1 = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_normal.jsonl'
file2 = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all.jsonl'
output_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all_with_normal.jsonl'

# 读取 jsonl 文件
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

# 合并两个文件内容
data1 = read_jsonl(file1)
data2 = read_jsonl(file2)

# 合并数据
merged_data = data1 + data2

# 自定义排序函数
def sort_key(item):
    # 排序规则：首先按 data_split 的顺序，然后按 video_name 字母顺序（normal 在最后）
    data_split_order = {'train': 0, 'val': 1, 'test': 2}
    video_name = item['video_name']
    
    # 如果 video_name 包含 "Normal" 字样，则排序时放在最后
    is_normal = 'Normal' in video_name
    return (data_split_order[item['data_split']], is_normal, video_name)

# 对合并后的数据进行排序
merged_data.sort(key=sort_key)

# 重新编号索引
for i, item in enumerate(merged_data):
    item['index'] = i

# 将合并和排序后的数据写回新文件
with open(output_file, 'w', encoding='utf-8') as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"合并并排序后的文件已保存至: {output_file}")