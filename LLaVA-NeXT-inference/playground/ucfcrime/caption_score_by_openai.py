import pandas as pd
import openai
from openai import OpenAI
import csv
import os
# 设置中转 API 基础地址和 API 密钥
client = OpenAI(
    # This is the default and can be omitted
    api_key='sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1' ,
    base_url="https://aihubmix.com/v1"
)
# 读取CSV文件
csv_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/video_caption_comparison_chinese.csv'
df = pd.read_csv(csv_file)

# 获取人工标注数据和其他标注数据
human_annotations = df[df['标注作者'] == '人工']
other_annotations = df[df['标注作者'] != '人工']

# 定义函数，发送API请求并评估两个文本的相似度
def evaluate_similarity(human_text, other_text):
    prompt = f"""
    Compare the ground truth from human and answers from AI models, to give a correctness score for models' performance. 
    Provide a score from 0 to 100, where 100 indicates AI answers contain all details of ground truth and have no unreasonable additional description.
    The two descriptions both try to summarize an abnormal event in a video. 
    You should evaluate the answer by 4 parts: 
    1. Whether AI accurately identify if there is an abnormal event in the video.
    2. Whether AI accurately summarize the type of abnormal behavior observed, for example, Abuse, Stealing, Fighting, Robbery, Arson.
    3. How many key event details mentioned in the human description are correctly captured by the AI.
    4. How many key event details in the AI's description are incorrect or inaccurate.
    
    Description 1 (Human): {human_text}
    
    Description 2 (AI): {other_text}
    
    Now give your score and reasons in form of {{"score":99, "reasons":[reason1,reason2,...]}}. The reasons must be in Chinese and is preferred te contain details.
    """

    response = client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are an expert evaluator tasked with assessing the accuracy and quality of AI-generated video descriptions. "},
                    {"role": "user", "content": prompt}
                ],
        model="gpt-3.5-turbo",
        )
    return response.choices[0].message.content

# 用于存储结果的列表
results = []

# 遍历所有相同video_name的描述
for video_name in human_annotations['video_name'].unique():
    human_text = human_annotations[human_annotations['video_name'] == video_name]['description'].values[0]
    other_text = other_annotations[other_annotations['video_name'] == video_name]['description'].values[0]
    
    # 调用OpenAI API比较文本
    similarity_score = evaluate_similarity(human_text, other_text)
    print(similarity_score)
    # 将结果保存
    results.append({
        'video_name': video_name,
        'human_description': human_text,
        'other_description': other_text,
        'similarity_score': similarity_score
    })

# 转换结果为DataFrame并保存为CSV
result_df = pd.DataFrame(results)
result_df.to_csv('/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/similarity_score_results.csv', index=False)
