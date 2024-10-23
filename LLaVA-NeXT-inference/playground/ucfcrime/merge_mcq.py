import json

def merge_jsonl_files(file_path1, file_path2, output_file_path):
    # 读取第一个JSONL文件并建立一个以index为键的字典
    data1 = {}
    with open(file_path1, 'r') as file:
        for line in file:
            data = json.loads(line)
            index = int(data['index']) # 假设每个JSON对象只有一个键值对，键为index
            data1[index] = data

    # 读取第二个JSONL文件并合并数据
    with open(file_path2, 'r') as file:
        for line in file:
            data = json.loads(line)
            index = int(data['index'])  # 假设每个JSON对象只有一个键值对，键为index
            if index in data1:
                # 如果index匹配，合并字典
                data1[index].update(data)
            else:
                print("Error: Index do not match.")

    # 将合并后的数据写入到新的JSONL文件中
    with open(output_file_path, 'w') as file:
        for index, value in data1.items():
            file.write(json.dumps( value)+'\n')

# 使用示例
merge_jsonl_files('/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_video_caption_ground_truth.jsonl',
 '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_MCQ_ground_truth_test.jsonl', 
 '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl')