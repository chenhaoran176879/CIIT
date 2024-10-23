import torch
import json
import torch
import os
import copy

from bench_hierarchical_questioning_detailed import anomaly_detection_question,event_description_question,crime_classification_question,temporal_grounding_question,event_description_with_classification,multiple_choice_question
class UCVLDataset(torch.utils.data.Dataset):
    # UCVL: UCF-Crime for Video Language models
    def __init__(self, json_root, video_folder,questions,split='train'):
        self.json_root = json_root
        self.video_folder = video_folder  # 根视频文件夹
        self.train_split = []
        self.val_split = []
        self.test_split = []
        self.split = split  # 用于选择要使用的分支（train, val, test）
        all_questions = {
            'anomaly_detection_question': anomaly_detection_question,
            'event_description_question': event_description_question,
            'crime_classification_question': crime_classification_question,
            'event_description_with_classification':event_description_with_classification,
            'temporal_grounding_question': temporal_grounding_question,
            'multiple_choice_question':multiple_choice_question
        }
        
        # 根据用户选择的 questions 构建 bench_questions
        if questions is None:
            questions = all_questions.keys()  # 默认选择所有问题
        self.bench_questions = {q: all_questions[q] for q in questions if q in all_questions}

        # 读取并加载 JSON 数据到不同的分支
        self.load_json()


    def load_json(self):
        # Load JSONL data into splits
        with open(self.json_root, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Append the data to the corresponding split
                if data['data_split'] == 'train':
                    self.train_split.append(data)
                elif data['data_split'] == 'val':
                    self.val_split.append(data)
                elif data['data_split'] == 'test':
                    self.test_split.append(data)

    def __len__(self):
        # 根据选定的分支返回对应数据的长度
        if self.split == 'train':
            return len(self.train_split)
        elif self.split == 'val':
            return len(self.val_split)
        elif self.split == 'test':
            return len(self.test_split)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, idx):
        # 根据选定的分支获取对应索引的数据
        if self.split == 'train':
            data = self.train_split[idx]
        elif self.split == 'val':
            data = self.val_split[idx]
        elif self.split == 'test':
            data = self.test_split[idx]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # 获取视频路径并将其添加到数据中
        video_name = data['video_name']
        video_path = self.get_video_path(video_name)  # 使用分类名构造路径
        # 增加 video_path 属性
        data['video_path'] = video_path
        
        bench_questions = copy.deepcopy(self.bench_questions)
        if 'temporal_grounding_question' in bench_questions:
            classification = self.get_category(data['video_name'])
            bench_questions['temporal_grounding_question'] = bench_questions['temporal_grounding_question'].format(classification)

        if 'event_description_with_classification' in bench_questions:
            classification = self.get_category(data['video_name'])
            bench_questions['event_description_with_classification'] = bench_questions['event_description_with_classification'].format(classification)        
            
        if 'multiple_choice_question' in bench_questions and 'multiple_choice_questions' in data:
            multiple_choice_questions = self.form_multiple_choice_questions(data['multiple_choice_questions'])
            for i in range(0,len(multiple_choice_questions)):
                bench_questions[f'multiple_choice_question_{i+1}'] = bench_questions['multiple_choice_question'].format(multiple_choice_questions[i])
            bench_questions.pop('multiple_choice_question')
        data['bench_questions'] = bench_questions    
        return data

    def get_video_path(self, video_name):
        # 简化的分类获取逻辑，遇到第一个数字就截取
        category = self.get_category(video_name)
        
        video_path = os.path.join(self.video_folder, category, video_name)
        if '.' not in video_path:
            video_path += '.mp4'
        return video_path
    
    def get_category(self, video_name):
        # 简化的分类获取逻辑，遇到第一个数字就截取
        for i, char in enumerate(video_name):
            if char.isdigit():
                return video_name[:i]
        return video_name  # 如果没有数字，返回完整名称
    
    def form_multiple_choice_questions(self,data):
        questions_text = []
        for key, value in data.items():
            question = value["question"]
            options = value["options"]
            text = f"Question {key}: {question}\n"
            text += "Options:\n"
            for option, description in options.items():
                text += f"  {option}: {description}\n"
            questions_text.append(text.strip())
        return questions_text



if __name__ == "__main__":
    json_root = "/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_combined.jsonl"
    video_folder = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/"
    dataset = UCVLDataset(json_root=json_root,video_folder=video_folder)
    print(dataset[0])

