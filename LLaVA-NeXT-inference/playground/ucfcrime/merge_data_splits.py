import json

# 定义文件路径
test_split = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_test_ordered.jsonl"
val_split = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_val_ordered.jsonl"
train_split = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_train_ordered.jsonl"

# 文件路径与 data_split 标签的对应关系
splits = [
    (train_split, "train"),
    (val_split, "val"),
    (test_split, "test")
]

# 用于存储合并后的数据
combined_data = []

# 处理每个 split 文件
for file_path, split in splits:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data['data_split'] = split
            combined_data.append(data)

# 重新排序 index
for i, entry in enumerate(combined_data):
    entry['index'] = i

# 将合并后的数据保存到新文件
output_file = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_combined.jsonl"
with open(output_file, 'w', encoding='utf-8') as out_f:
    for entry in combined_data:
        out_f.write(json.dumps(entry) + "\n")

print(f"合并并重新排序的数据已保存到 {output_file}")
