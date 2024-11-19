import os
import re
import json
from copy import deepcopy
def merge_answers(file1, file2, output_file):
    merged_data = []
    
    # 读取第一个 JSONL 文件
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = [json.loads(line) for line in f1]
    if not data1 or len(data1)<200:
        print(f"skipping {file1}")
        return

    
    # 读取第二个 JSONL 文件
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = [json.loads(line) for line in f2]
    
    if not data2 or len(data2)<200:
        print(f"skipping {file1}")
        return
 
    
    # 确保两个文件中的数据条数相同
    assert len(data1) == len(data2), "The two files must have the same number of entries."

    # 遍历数据，合并 answer 字段
    for entry1, entry2 in zip(data1, data2):
        assert entry1['index'] == entry2['index'], "Indexes must match."
        assert entry1['video_name'] == entry2['video_name'], "Video names must match."
        
        # 合并 answer 字段
        merged_answers = deepcopy(entry1)
        merged_answers['answer']['event_description_with_classification'] = entry2['answer']['event_description_with_classification']

        
        # # 创建合并后的条目
        # merged_entry = {
        #     "index": entry1['index'],
        #     "data_split": entry1['data_split'],
        #     "model_name": entry1['model_name'],
        #     "video_name": entry1['video_name'],
        #     "nframes": entry1['nframes'],
        #     "answer": merged_answers
        # }
        
        merged_data.append(merged_answers)
    
    # 将合并后的数据写入新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in merged_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def batch_process(result_path):
    # 正则匹配模式
    pattern1 = re.compile(r'eval_results_(.*?)_32frames_Description\.jsonl')
    #pattern2 = re.compile(r'eval_results_(.*?)_32frames_MCQ_TEST\.jsonl')
    
    # 获取文件列表
    file_paths = os.listdir(result_path)

    # 遍历文件，找到匹配的文件
    for file1 in file_paths:
        match1 = pattern1.match(file1)
        if match1:
            # 从第一个文件中提取通用标识符
            common_identifier = match1.group(1)
            
            # 根据通用标识符构造第二个文件的名称
            expected_file2 = f"eval_results_{common_identifier}_32frames_Description_with_class_only_1023.jsonl"
            
            # 检查第二个文件是否存在
            if expected_file2 in file_paths:
                # 构造输出文件名
                output_file = os.path.join(result_path, f"eval_results_{common_identifier}_32frames_merge_1023.jsonl")
                
                # 获取文件的完整路径
                file1_path = os.path.join(result_path, file1)
                file2_path = os.path.join(result_path, expected_file2)
                
                # 合并文件并输出结果
                print(f"Merging {file1} and {expected_file2} into {output_file}")
                merge_answers(file1_path, file2_path, output_file)
            else:
                print(f"Matching file for {file1} not found.")

# 运行批处理函数
result_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/"
batch_process(result_path)