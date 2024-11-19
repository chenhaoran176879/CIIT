# 将实验结果分析成几张表格
# 1.  纵坐标： 8个模型；横坐标：不同题目的得分，以及总分
# 2.  纵坐标： 不同犯罪分类 横坐标：8个模型的得分（最好是分主观题和客观题）
# 3.  纵坐标： 8个模型； 横坐标： 不同帧率下的总分
import os
import json
import re
import pandas as pd


def score_summary_per_question(base_dir='/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/', output_file='collected_data_32frames.csv'):
    """
    收集指定目录下的JSONL文件中的数据，并将其保存为CSV文件，模型名和nframe放在第一列，并按模型字典顺序保存简称。
    允许自定义列顺序。
    
    :param base_dir: 包含JSONL文件的目录路径
    :param output_file: 导出的CSV文件名
    """

    # 文件名的正则表达式模式，提取模型名和nframe
    file_pattern = re.compile(r'eval_results_(.*?)_(\d+)frames_merge_1023_score_1023\.jsonl')

    # 模型名到简称的映射字典，定义排序顺序
    model_order_dict = {
        'OpenGVLab--InternVL2-1B': 'InternVL2-1B',
        'qwen-vl2-2B': 'qwen-VL2-2B',
        'Qwen2-VL-7B-Instruct': 'qwen-VL2-7B',
        'lmms-lab--llava-onevision-qwen2-7b-ov':'llava-ov-qwen7B',
        'OpenGVLab--InternVL2-8B':'InternVL2-8B',
        'OpenGVLab--InternVL2-26B':'InternVL2-26B',
        'OpenGVLab--InternVL2-40B':'InternVL2-40B',
        # 其他模型按需添加
    }

    # 预定义的数据列顺序
    desired_column_order = ['model_name', 'average_sum','nframe','anomaly_detection_question','event_description_with_classification','crime_classification_question','multiple_choice_total','openai_model']  # 按需修改

    # 创建一个存储结果的列表
    collected_data = []

    # 遍历基准目录下的所有文件
    for file_name in os.listdir(base_dir):
        match = file_pattern.match(file_name)
        print(match)
        if match:
            # 提取模型名和nframe
            model_name = match.group(1)
            nframe = match.group(2)  # 提取帧数
            if int(nframe) != 32:
                continue
            
            # 检查是否在字典中，如果不在，跳过此模型
            if model_name not in model_order_dict:
                
                continue
            
            # 获取简称
            model_short_name = model_order_dict[model_name]
            
            file_path = os.path.join(base_dir, file_name)
            print(f"处理文件: {file_name}, 模型名: {model_short_name}, nframe: {nframe}")

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 去除空行
            lines = [line.strip() for line in lines if line.strip()]

            if lines:
                # 处理倒数第二行或最后一行的情况
                last_line = lines[-1]
                try:
                    json_data = json.loads(last_line)
                except json.JSONDecodeError:
                    # 如果最后一行无效，尝试倒数第二行
                    last_line = lines[-2]
                    json_data = json.loads(last_line)

                # 将简称和nframe添加到数据中
                json_data["model_name"] = model_short_name
                json_data["nframe"] = nframe

                # 合并5个multiple_choice_question分数为一个总分
                mc_score_keys = [f'multiple_choice_question_{i+1}' for i in range(5)]
                mc_total_score = sum(json_data.get(key, 0) for key in mc_score_keys)
                json_data['multiple_choice_total'] = round(mc_total_score, 1)  # 保留一位小数

                # 删除原来的multiple_choice_question分数
                for key in mc_score_keys:
                    if key in json_data:
                        del json_data[key]

                # 将数据添加到收集列表中
                collected_data.append(json_data)
            else:
                print(f"文件 {file_name} 为空或无效")

    # 将收集的数据转换为pandas DataFrame
    df = pd.DataFrame(collected_data)
    print(df)

    # 确保'model_name'和'nframe'作为前两列
    columns_order = ['model_name', 'nframe'] + [col for col in df.columns if col not in ['model_name', 'nframe']]

    # 按照模型名的字典顺序排序
    df['model_sort'] = df['model_name'].map(lambda x: list(model_order_dict.values()).index(x))
    df.sort_values(by='model_sort', inplace=True)
    df.drop(columns=['model_sort'], inplace=True)

    # 自定义列顺序，保留已定义列
    columns_in_order = [col for col in desired_column_order if col in df.columns]
    df = df[columns_in_order + [col for col in df.columns if col not in columns_in_order]]

    # 保留所有数值列的小数点后一位
    df = df.round(1)

    # 将数据写入CSV文件
    df.to_csv(os.path.join(base_dir, output_file), index=False)

    print(f"数据已成功写入 {output_file}")
    
def score_summary_per_crime():
    return


def score_summary_per_nframes():
    # qwen-vl2-72B_64frames CUDA out of memory
    return


import pandas as pd

def csv_to_latex_table(csv_file, latex_file='/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/table.tex'):
    """
    将CSV文件转换为LaTeX表格并输出到指定文件。

    :param csv_file: 输入的CSV文件路径
    :param latex_file: 输出的LaTeX文件名
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 生成LaTeX表格
    latex_code = df.to_latex(index=False, escape=False, column_format='|l|' + 'l|' * (df.shape[1] - 1))

    # 写入到LaTeX文件
    with open(latex_file, 'w') as f:
        f.write(latex_code)

    print(f"LaTeX表格已成功写入 {latex_file}")

# 使用示例
#csv_to_latex_table('/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/collected_data.csv')


score_summary_per_question()