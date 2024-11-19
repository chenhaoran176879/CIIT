import os
import re
import json
import pandas as pd
from collections import defaultdict

# 定义目标路径
directory = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/"

# 正则表达式匹配文件名：前缀_数字.jsonl
pattern = re.compile(r"^(.*)_(\d{6})\.jsonl$")

# 用字典按前缀存储文件的 event_description_question
grouped_results = defaultdict(dict)

# 遍历文件夹
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        prefix = match.group(1)
        file_index = match.group(2)  # 提取文件的数字部分
        filepath = os.path.join(directory, filename)
        
        # 读取文件的最后一行（非空白行）
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                last_line = lines[-1]
                try:
                    data = json.loads(last_line)  # 解析 JSON 数据
                    event_question = data.get("event_description_question", "")
                    grouped_results[prefix][file_index] = event_question
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

# 将结果转换为 DataFrame
rows = []
for prefix, files in grouped_results.items():
    row = {"Prefix": prefix}
    for file_index, event_question in sorted(files.items()):
        row[f"File_{file_index}"] = event_question
    rows.append(row)

df = pd.DataFrame(rows)

# 打印表格
print(df)

# 保存为 CSV 文件
output_file = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/merged_event_description_questions.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Results saved to {output_file}")
