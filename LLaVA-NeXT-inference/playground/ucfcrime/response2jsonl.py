import json
from collections import OrderedDict

def load_jsonl(filepath):
    """从JSONL文件中逐行读取数据"""
    with open(filepath, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def convert_response_to_dict(data):
    """将response字符串转为有序字典，并将index提前"""
    if 'response' in data:
        response_str = data['response']
        index = data['index']
        try:
            # 尝试将字符串转换为字典
            response_dict = json.loads(response_str)
            
            # 创建一个有序字典，并将index放在第一位
            ordered_response_dict = OrderedDict()
            ordered_response_dict['index'] = index
            
            # 将原始response字典的其他键按顺序插入
            for key, value in response_dict.items():
                ordered_response_dict[key] = value
                
            return ordered_response_dict
        except json.JSONDecodeError:
            print(f"无法解析response为字典: {response_str}")
            return None
    return None

def process_jsonl(input_filepath, output_filepath):
    """读取JSONL文件，转换response，并将结果存储到新文件中"""
    dataset = load_jsonl(input_filepath)

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for data in dataset:
            response_dict = convert_response_to_dict(data)
            if response_dict:
                # 将有序字典写入文件
                outfile.write(json.dumps(response_dict, ensure_ascii=False) + '\n')
            else:
                print("response不是有效的字典字符串。")

# 使用你的JSONL文件路径
input_jsonl_file_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_val.jsonl"
output_jsonl_file_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_val_ordered.jsonl"

# 处理并存储结果
process_jsonl(input_jsonl_file_path, output_jsonl_file_path)
