import pandas as pd
import openai
from openai import OpenAI
import csv
import os
from bench_hierarchical_questioning import  eval_event_description_prompt,eval_event_description_with_classification_prompt
import json
from copy import deepcopy
import argparse
from video_utils import load_last_json_index


def convert_response_to_numeric(value):
    try:
        # 如果包含小数点或是科学计数法的符号，转换为浮点数
        if '.' in value or 'e' in value.lower():
            return float(value)
        else:
            # 如果没有小数点且是有效数字，转换为整数
            return int(value)
    except ValueError:
        number = ''
        for char in value:
            if char.isdigit():
                number += char  # 拼接数字字符
            elif number:  # 如果已经开始拼接数字且遇到非数字字符
                break
        print(f"response is expected to be number, but got {value}. Force convert to {number}")
        return int(number)
    
class OpenAIChatBot:
    def __init__(self):
        self.client = OpenAI(
            api_key='sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1' ,
            base_url="https://aihubmix.com/v1"
            )
        self.model_type = "gpt-3.5-turbo"
    def chat(self,message):
        response = self.client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are a professional grader. \
                     Your task is to evaluate the given answers strictly based on the provided rules and assign a numeric score. \
                     Do not provide any explanations, just the score. "},
                    {"role": "user", "content": message}
                ],
        model=self.model_type,
        )
        text = response.choices[0].message.content
        
        return text
    

class UCVLAnswerDataset:
    def __init__(self,answer_path, ground_truth_path):
        self.eval_prompts = {
            "event_description_question":eval_event_description_prompt,
            "event_description_with_classification":eval_event_description_with_classification_prompt
        }
        self.data = self.load_data(answer_path)
        self.ground_truth = self.load_ground_truth(ground_truth_path)


    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data


    def load_ground_truth(self,file_path):
        data_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                video_name = data.get('video_name')
                if video_name:
                    data_dict[video_name] = data
        return data_dict


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        answers = self.data[index]
        video_name = answers['video_name']
        gt = self.ground_truth[video_name]

        return {
                "model_answer":answers,
                "ground_truth":gt
                }


def get_category(video_name):
        # 简化的分类获取逻辑，遇到第一个数字就截取
        for i, char in enumerate(video_name):
            if char.isdigit():
                return video_name[:i]
        return video_name  # 如果没有数字，返回完整名称


def calculate_averages(data_list):
    if not data_list:
        return {}

    # 初始化一个字典来存储每个字段的总和
    field_sums = {}
    # 初始化一个字典来存储每个字段的数量
    field_counts = {}

    # 遍历列表中的每个字典
    for data in data_list:
        for field, value in data.items():
            # 如果字段不在字典中，初始化它
            if field not in field_sums:
                field_sums[field] = 0
                field_counts[field] = 0
            # 累加字段的值
            field_sums[field] += value
            # 增加字段的计数
            field_counts[field] += 1

    # 计算每个字段的平均值
    averages = {}
    for field in field_sums:
        averages[field] = field_sums[field] / field_counts[field]

    # 计算所有字段的总平均值
    total_sum = sum(averages.values())
    total_fields = len(averages)
    overall_average = total_sum / total_fields if total_fields > 0 else 0

    # 将总平均值添加到字典中
    averages['overall_average'] = overall_average

    return averages


def sum_score(scores:dict)->float:
    return (
    30 *  (scores.get('anomaly_detection_question',0)>0) +
    30 * (scores.get('crime_classification_question',0)>0) + # bonus
    30 * (scores.get('event_description_question',0)/100) +
    40 * (scores.get('event_description_with_classification',0)/100))


def UCVL_scoring(args):
    save_score_path = args.save_score_path
    model_name = None
    client = OpenAIChatBot()
    model_answer_dataset = UCVLAnswerDataset(
        answer_path=args.answer_path,
        ground_truth_path=args.ground_truth_path)
    start_index = load_last_json_index(save_score_path)+1
    f =  open(save_score_path,'a',encoding='utf-8')
    final_scores = []
    for idx in range(start_index, len(model_answer_dataset)):
        answer_sheet = model_answer_dataset[idx]
        if model_name == None: model_name = answer_sheet['model_answer']['model_name']
        scores = {}
        for question_name,model_answer in answer_sheet['model_answer']['answer'].items():
            eval_prompt_format = model_answer_dataset.eval_prompts.get(question_name,None)
            if eval_prompt_format:
                eval_prompt = eval_prompt_format.format(answer_sheet['ground_truth']['description'],model_answer)
                score = client.chat(eval_prompt)
                score = convert_response_to_numeric(score)
                scores[question_name] = score
                
                print(f"问题: {question_name}")
                print(f"模型回答: {model_answer}")
                print(f"ground truth: 分类{answer_sheet['model_answer']['video_name']}, \n ground truth:描述{answer_sheet['ground_truth']['description']}")
                print(f"得分: {score}")

            elif question_name == "anomaly_detection_question":
                score = 100
                prefix = model_answer[:min(len(model_answer),5)]
                gt = None
                if 'normal' in  answer_sheet['ground_truth']['video_name'].lower():
                    gt = 0
                else: gt = 1

                if 'yes' in prefix.lower():
                    score *= gt

                elif 'no' in prefix.lower():
                    score *= (1-gt)

                elif 'yes' in model_answer.lower() and 'no' not in model_answer.lower():
                    score *= gt
                elif 'yes' not in model_answer.lower() and 'no' in model_answer.lower():
                    score *= gt
                
                else:
                    score = 0
            
                scores[question_name] = score

            elif question_name == "crime_classification_question":
                score = 0
                if get_category(answer_sheet['ground_truth']['video_name']).lower() in model_answer.lower():
                    score = 100
                scores[question_name] = score

        result = deepcopy(answer_sheet["model_answer"])
        result['scores'] =scores
        score_sum = sum_score(scores)
        print(f"本视频最终平均得分: {score_sum}")
        result['sum'] = score_sum
        scores['sum'] = score_sum
        final_scores.append(scores)
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    

    averages = calculate_averages(final_scores)
    averages['model_name'] = model_name
    averages['openai_model'] = client.model_type
    print("Averages:\n",averages)
    if model_name:
        f.write(json.dumps(averages, ensure_ascii=False) + '\n')
    f.close()
          
    return



if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Process evaluation paths.')
    parser.add_argument('--save_score_path', type=str, default=None, help='Path to save the score results')
    parser.add_argument('--answer_path', type=str, default=None, help='Path to the model answer JSONL file')
    parser.add_argument('--ground_truth_path', type=str, default=None, help='Path to the ground truth JSONL file')

    args = parser.parse_args()

    answer_paths = [
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_Qwen2-VL-7B-Instruct_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_OpenGVLab--InternVL2-1B_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_qwen-vl2-72B_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_OpenGVLab--InternVL2-40B_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_qwen-vl2-2B_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_lmms-lab--llava-onevision-qwen2-7b-ov_32frames_1013.jsonl",
        "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/eval_results_OpenGVLab--InternVL2-1B_32frames_1013.jsonl",
        
    ]

    args.ground_truth_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_combined.jsonl"
    for answer_path in answer_paths:
        args.answer_path = answer_path
        args.save_score_path = answer_path.split('.')[0]+'_score.jsonl'
        print(args)
        UCVL_scoring(args)
        exit(0)