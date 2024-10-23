import json

# 文件路径
file_with_split = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all_with_normal.jsonl'
file_without_split = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all_with_normal_1016_eng.jsonl'
output_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all_with_normal_1016_eng_updated.jsonl'

# 读取有 data_split 的文件
data_with_split = {}
with open(file_with_split, 'r') as f:
    for line in f:
        item = json.loads(line)
        video_name = item.get('video_name')
        data_split = item.get('data_split')
        if video_name and data_split:
            # 去掉文件扩展名，以匹配第二个文件中的 video_name
            video_name = video_name.split('.')[0]
            data_with_split[video_name] = data_split

# 更新没有 data_split 的文件
updated_data = []
with open(file_without_split, 'r') as f:
    for line in f:
        item = json.loads(line)
        video_name = item.get('video_name', '').split('.')[0]  # 去掉文件扩展名
        # 如果存在对应的 data_split，则添加
        if video_name in data_with_split:
            item['data_split'] = data_with_split[video_name]
        updated_data.append(item)

# 将更新后的数据写入新的文件
with open(output_file, 'w') as f:
    for item in updated_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"更新完成，结果保存在 {output_file}")