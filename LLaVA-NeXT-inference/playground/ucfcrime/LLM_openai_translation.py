import openai
from openai import OpenAI
import csv
import os
import re
import json

# 设置中转 API 基础地址和 API 密钥
openai.api_base = "https://aihubmix.com/v1"  # 中转 API 基础地址
openai.api_key = 'sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1'  # 中转 API 密钥

# 读取CSV文件
input_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/video_caption_comparison.csv'
output_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/video_caption_comparison_chinese.csv'



def contains_chinese(text):
    # 匹配汉字的正则表达式范围：\u4e00-\u9fff
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(text))


def translate_chn2eng(text):
    # 确保你已经设置了你的API密钥
    client = OpenAI(
        # This is the default and can be omitted
        api_key='sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1' ,
        base_url="https://aihubmix.com/v1"
    )

    response = client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are a professional translator. Please translate a mixture of Chinese and English to ONLY English and keep the original data form."},
                    {"role": "user", "content": text}
                ],
        model="gpt-3.5-turbo",
        )
    translation = response.choices[0].message.content
    return translation


def translate_eng2chn(text):
    # 确保你已经设置了你的API密钥



    client = OpenAI(
        # This is the default and can be omitted
        api_key='sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1' ,
        base_url="https://aihubmix.com/v1"
    )

    response = client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are a professional translator. Please translate the following text from English to Chinese."},
                    {"role": "user", "content": text}
                ],
        model="gpt-3.5-turbo",
        )
    translation = response.choices[0].message.content
    print(translation)
    return translation


def eng2chn():
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if row['标注作者'] == 'LLM' and row['description']:  # 只翻译LLM的英文描述
                row['description'] = translate_eng2chn(row['description'])
            writer.writerow(row)

    print("翻译已完成并存储到新CSV文件中。")



def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data




def remaining_chn2eng():
    input_bilingual = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results.jsonl"
    output_eng = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_pure_eng.jsonl"

    dataset = load_json(input_bilingual)
    with open(output_eng, mode='w', newline='', encoding='utf-8') as outfile:
        for data in dataset:
            response = data['response']
            if contains_chinese(response):
                print(response)
                response = translate_chn2eng(response)
                data['response'] = response
                print(response)
                outfile.write(json.dumps(data) + '\n')

remaining_chn2eng()