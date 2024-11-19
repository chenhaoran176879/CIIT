from openai import OpenAI
import re
import json
from UCVLDataset import UCVLDataset
import os
import re


def parse_mcq_from_text(text):
    # Regular expressions to match each question and its associated options and ground truth
    mcq_pattern = re.compile(r'\[MCQ_(\d+)\]\s*(.*?)\s*\[A\]\s*(.*?)\s*\[B\]\s*(.*?)\s*\[C\]\s*(.*?)\s*\[D\]\s*(.*?)\s*\[ground_truth\]\s*(\w)', re.DOTALL)

    # Initialize the dictionary to store multiple-choice questions
    mcq_dict = {}

    # Find all matches in the text using the compiled pattern
    matches = mcq_pattern.findall(text)

    # Loop through each match to build the mcq_dict
    for match in matches:
        q_id = match[0]  # Question ID
        question = match[1].strip()  # Question text
        option_a = match[2].strip()  # Option A
        option_b = match[3].strip()  # Option B
        option_c = match[4].strip()  # Option C
        option_d = match[5].strip()  # Option D
        ground_truth = match[6].strip()  # Ground truth answer

        # Add the question and options to the dictionary
        mcq_dict[q_id] = {
            "question": question,
            "options": {
                "A": option_a,
                "B": option_b,
                "C": option_c,
                "D": option_d,
            },
            "ground_truth": ground_truth
        }

    return mcq_dict



class OpenAIChatBot:
    def __init__(self):
        self.client = OpenAI(
            api_key='sk-6LF1pPRgAAaakNvGB2Ce48550bDe4f5099DfEd2a62771aE1' ,
            base_url="https://aihubmix.com/v1"
            )
        self.model_type = "gpt-3.5-turbo"
    def chat(self,message):
        response = self.client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are an expert in generating multiple-choice questions (MCQs) based on detailed video description."},
                    {"role": "user", "content": message}
                ],
        model=self.model_type,
        )
        text = response.choices[0].message.content
        
        return text

def get_category(video_name):
    # 简化的分类获取逻辑，遇到第一个数字就截取
    for i, char in enumerate(video_name):
        if char.isdigit():
            return video_name[:i]
    return video_name  # 如果没有数字，返回完整名称

MCQ_design_prompt="""Here is a description of a video: {}.
The video's name is {} which denotes human definition of the main event in this video.
Please prepare five multiple choice questions(MCQ) with 4 choices (A,B,C,D) for this video.
You should consider these aspects:
1. The course of main event;
2. The main features of the environment and the participants(if there are. Objects can also be participants);
3. What the participants do.
4. What is the subsequence of this event.

You can use one aspect to make more than one question. Just choose what you think proper for this description.
You should avoid hinting at the answer to one question in the stem or options of other questions.
You should also avoid asking similar questions or set similar options between questions.
Every option(A,B,C,D) has 25% probability to be ground truth.

Question example(keep strict to this format):

[MCQ_1]What are the characteristics of the location where the assault occurred?
[A]On an empty street with no one around.
[B]In a crowded bar with many people nearby.
[C]In a quiet corner of a restaurant with only a few customers.
[D]In a parking lot with only a few cars nearby.
[ground_truth]B


Now please make up 5 questions, each with 4 options and 1 ground truth:
"""

def main():
    # Initialize the dataset
    dataset = UCVLDataset(
        json_root="/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl",
        video_folder="/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/",
        split='val',
        questions=None
    )

    print(len(dataset))
    MCQ_save_path = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_val_mcq.jsonl'
    client = OpenAIChatBot()

    # Load existing indices to skip
    existing_indices = set()
    if os.path.exists(MCQ_save_path):
        with open(MCQ_save_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'index' in data:
                    existing_indices.add(data['index'])

    with open(MCQ_save_path, 'a', encoding='utf-8') as f:
        for index in range(0, len(dataset)):
            # Skip existing indices
            data_split_index = index+1158 # if test: index+1536 if val: index+1158
            if data_split_index in existing_indices:
                print('skipping ', data_split_index)
                continue
            
            data = dataset[index]
            video_category = get_category(data['video_name'])
            description = data['description']
            response = client.chat(MCQ_design_prompt.format(description, video_category))
            print(response)
            MCQ_dict = parse_mcq_from_text(response)

            # Check for empty values in MCQ_dict
            if any(not value for value in MCQ_dict.values() if isinstance(value, (dict, str))):
                raise ValueError(f"MCQ_dict contains empty values for index {data_split_index}: {MCQ_dict}")

            # Clean up the data and add MCQ_dict
            data.pop('video_path', None)
            data.pop('bench_questions', None)
            data['multiple_choice_questions'] = MCQ_dict
            data['index'] = data_split_index  # Ensure to add the index to data for reference

            f.write(json.dumps(data) + '\n')
            print(json.dumps(data))

# Call the main function
if __name__ == "__main__":
    main()
  