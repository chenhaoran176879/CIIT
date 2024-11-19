import pandas as pd
import openai
from openai import OpenAI
import csv
import os
import logging
from bench_hierarchical_questioning_detailed import  eval_event_description_prompt,eval_event_description_with_classification_prompt,eval_event_description_prompt_uca
import json
from copy import deepcopy
import argparse
from video_utils import load_last_json_index


def convert_response_to_numeric(value):
    if not isinstance(value,str):
        value = str(value)
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
        logger.info(f"response is expected to be number, but got {value}. Force convert to {number}")
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

        self.eval_prompts_uca = {
            "event_description_question":eval_event_description_prompt_uca,
            #"event_description_with_classification":eval_event_description_with_classification_prompt
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
        if '.' not in video_name:
            video_name+='.mp4'
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
    # total_fields = len(averages)
    # overall_average = total_sum / total_fields if total_fields > 0 else 0

    # 将总平均值添加到字典中
    averages['average_sum'] = total_sum

    return averages


def sum_score(scores:dict)->float:
    return (
    10 *  (scores.get('multiple_choice_question_1',0)>0) +
    10 *  (scores.get('multiple_choice_question_2',0)>0) +
    10 *  (scores.get('multiple_choice_question_3',0)>0) +
    10 *  (scores.get('multiple_choice_question_4',0)>0) +
    10 *  (scores.get('multiple_choice_question_5',0)>0) +
    40 *  (scores.get('anomaly_detection_question',0)>0) +
    20 * (scores.get('crime_classification_question',0)>0) + # bonus
    30 * (scores.get('event_description_question',0)/100) +
    30 * (scores.get('event_description_with_classification',0)/100))

def extract_ABCD(text):
    if ':' in text:
        text = text.split(':')[0]
    
    if '.' in text:
        text = text.split('.')[0]
    text = text.lower().strip()
    return text 

def UCVL_scoring(args):
    save_score_path = args.save_score_path
    model_name = None
    client = OpenAIChatBot()
    model_answer_dataset = UCVLAnswerDataset(
        answer_path=args.answer_path,
        ground_truth_path=args.ground_truth_path)
    start_index,final_scores = load_last_json_index(save_score_path,collect_scores = True)
    print('continue from index',start_index)
    f =  open(save_score_path,'a',encoding='utf-8')
    #final_scores = []
    for idx in range(start_index, len(model_answer_dataset)):
        answer_sheet = model_answer_dataset[idx]
        if model_name == None: model_name = answer_sheet['model_answer']['model_name']
        scores = {}
        ground_truth_description= answer_sheet['ground_truth']['description']
        detection_score = 100
        result = deepcopy(answer_sheet["model_answer"])
        logger.info(f"视频名称：{answer_sheet['model_answer']['video_name']}")


        for question_name,model_answer in answer_sheet['model_answer']['answer'].items():
            
            eval_prompt_format = model_answer_dataset.eval_prompts.get(question_name,None)
            if isinstance(model_answer, list) and len(model_answer) == 1 : model_answer = model_answer[0]
            if eval_prompt_format:
                eval_prompt = eval_prompt_format.format(ground_truth_description,model_answer)
                try:
                    score = client.chat(eval_prompt)
                except Exception as e:
                    print(f"Error: model_name: {model_name} data index:{idx},Error info:{e}")
                    score = 0
                score = convert_response_to_numeric(score)
                scores[question_name] = score

                logger.info(f"{answer_sheet['model_answer']['video_name']}主观题问题: {eval_prompt}")
                logger.info(f"{answer_sheet['model_answer']['video_name']}主观题得分: {score}")

            elif question_name == "anomaly_detection_question":
                detection_score = 100
                prefix = model_answer[:min(len(model_answer),5)]
                gt = None
                if 'normal' in  answer_sheet['ground_truth']['video_name'].lower():
                    gt = 0
                else: gt = 1
                if 'yes' in prefix.lower():
                    detection_score *= gt

                elif 'no' in prefix.lower():
                    detection_score *= (1-gt)

                elif 'yes' in model_answer.lower() and 'no' not in model_answer.lower():
                    detection_score *= gt
                elif 'yes' not in model_answer.lower() and 'no' in model_answer.lower():
                    detection_score *= gt
                
                else:
                    detection_score = 0
            
                scores[question_name] = detection_score
                logger.info(f"{answer_sheet['model_answer']['video_name']}异常检测题： 回答{model_answer} 标准答案{bool(gt)}")

            elif question_name == "crime_classification_question":
                classification_score = 0
                if 'normal' in  answer_sheet['ground_truth']['video_name'].lower():
                    classification_score = detection_score
                if get_category(answer_sheet['ground_truth']['video_name']).lower() in model_answer.lower():
                    classification_score = detection_score
                scores[question_name] = classification_score
                logger.info(f"{answer_sheet['model_answer']['video_name']}分类题： 回答{model_answer} 标准答案{get_category(answer_sheet['ground_truth']['video_name']).lower()}")

            elif "multiple_choice_question" in question_name:
                question_index = str(question_name[-1])
                MCQ_gt_dict =  answer_sheet['ground_truth']['multiple_choice_questions']
                MCQ_gt = MCQ_gt_dict[question_index]['ground_truth']
                result[f'ground_truth_{question_name}'] = MCQ_gt
                if MCQ_gt.lower()==extract_ABCD(model_answer):
                    scores[question_name] = 100
                else:
                    scores[question_name] = 0

                logger.info(f"{answer_sheet['model_answer']['video_name']}选择题：\n \
                问题{MCQ_gt_dict[question_index]['question']}\n \
                A. {MCQ_gt_dict[question_index]['options']['A']}  \n \
                B. {MCQ_gt_dict[question_index]['options']['B']}  \n \
                C. {MCQ_gt_dict[question_index]['options']['C']}  \n \
                D. {MCQ_gt_dict[question_index]['options']['D']}  \n \
                回答{model_answer}   标准答案{extract_ABCD(model_answer)}")


        
        result['ground_truth_description'] = ground_truth_description            
        result['scores'] =scores
        score_sum = sum_score(scores)
        logger.info(f"本视频最终得分: {score_sum}/170")
        result['sum'] = score_sum
        scores['sum'] = score_sum
        final_scores.append(scores)
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    

    averages = calculate_averages(final_scores)
    averages['model_name'] = model_name
    averages['openai_model'] = client.model_type
    logger.info("Averages:\n",averages)
    if model_name:
        f.write(json.dumps(averages, ensure_ascii=False) + '\n')
    f.close()
          
    return

logger = None

def setup_logger(log_file, level=logging.INFO):
    """Function to set up the global logger"""
    global logger  # 使用全局 logger 变量
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger('main_logger')  # 给全局 logger 一个名称
    logger.setLevel(level)
    logger.addHandler(handler)
    
    if not logger.handlers:
        logger.addHandler(handler)


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Process evaluation paths.')
    parser.add_argument('--save_score_path', type=str, default=None, help='Path to save the score results')
    parser.add_argument('--answer_path', type=str, default=None, help='Path to the model answer JSONL file')
    #parser.add_argument('--MCQ_answer_path', type=str, default=None, help='Path to the model MCQ answer JSONL file'))
    parser.add_argument('--ground_truth_path', type=str, default=None, help='Path to the ground truth JSONL file')

    args = parser.parse_args()

    answer_paths = [
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_lmms-lab--llava-onevision-qwen2-7b-ov_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_OpenGVLab--InternVL2-1B_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_OpenGVLab--InternVL2-8B_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_OpenGVLab--InternVL2-26B_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_OpenGVLab--InternVL2-40B_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_qwen-vl2-2B_64frames_all_1023.jsonl",
    "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_Qwen2-VL-7B-Instruct_64frames_all_1023.jsonl",
    ]

    args.ground_truth_path = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl"

    # 新的结果保存路径
    save_dir = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/benchmark_results/"
    log_dir = save_dir
    for answer_path in answer_paths:
        args.answer_path = answer_path
        # 只取文件名部分，并替换路径为新的结果目录
        base_filename = os.path.basename(answer_path).split('.')[0]
        log_filename = os.path.join(log_dir, base_filename + '.log')
        args.save_score_path = os.path.join(save_dir, base_filename + '_score_uca_1118_003.jsonl')

        setup_logger(log_filename)

        # 打印和记录输出
        print(f"Processing {answer_path}")
        logger.info(f"Processing {answer_path}")
        print(args)
        logger.info(args)


        UCVL_scoring(args)

        logger.info('\n\n\n\n\n\n\n\n')

# python -u /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/bench_score_by_openai.py >> /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_lmms-lab--llava-onevision-qwen2-7b-ov_32frames_score_1016.log 2>&1 &