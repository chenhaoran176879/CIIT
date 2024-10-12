from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from video_summarization_prompt import video_summarization_prompt,video_summarization_prompt_normal

import json

device = "cuda" # the device to load the model onto



file_path = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/UCAtxt/UCFCrime_Val.txt"


class UCADataset:
    def __init__(self, file_path):
        self.data = []
        self._load_data(file_path)
    
    def _load_data(self, file_path):
        data_dict = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                filename, description = parts
                data_dict[filename].append(description)
        self.data = [(filename, descriptions) for filename, descriptions in data_dict.items()]

    def __getitem__(self, index):
        if 0 <= index < len(self.data):
            return self.data[index]
        else:
            print("索引超出范围，请输入有效的索引值。")
            return None
    def __len__(self):
        return len(self.data)






def load_processed_videos(output_file):
    """
    读取已经处理过的视频信息并返回一个集合，集合中存储视频文件名或者索引
    """
    processed_videos = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                processed_videos.add(result["video_name"])
                # 如果你希望用索引来记录，使用 result["index"] 替换 result["video_name"]
    except FileNotFoundError:
        print(f"{output_file} 文件不存在，将从头开始处理。")
    except Exception as e:
        print(f"读取 {output_file} 时发生错误: {e}")
    return processed_videos
import json

def load_processed_videos(output_file):
    processed_videos = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                processed_videos.add(result["video_name"])
    except FileNotFoundError:
        print(f"{output_file} 文件不存在，将从头开始处理。")
    except Exception as e:
        print(f"读取 {output_file} 时发生错误: {e}")
    return processed_videos


if __name__ == '__main__':
    dataset = UCADataset(file_path=file_path)
    print(f"Dataset loaded. Length: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        "/home/share/chenhaoran/model_zoo/Qwen1.5-72B-Chat-GPTQ-Int4/",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("/home/share/chenhaoran/model_zoo/Qwen1.5-72B-Chat-GPTQ-Int4/")
    device = model.device

    output_file = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/LLM_summarization_results_val_normal.jsonl'

    processed_videos = load_processed_videos(output_file)
    print(f"已处理视频数量: {len(processed_videos)}")

    with open(output_file, 'a') as f:
        for i in range(0, len(dataset)):
            try:
                data = dataset[i]
                video_name = data[0]

                if video_name in processed_videos:
                    print(f"Skipping already processed video: {video_name}")
                    continue

                if  not video_name.lower().startswith('normal'):
                    print(f"Skipping normal data at index {i}: {video_name}")
                    continue

                print(f"Processing index {i}, video name: {video_name}")

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": video_summarization_prompt_normal.format(str(data))}
                ]

                print(f"Messages for index {i}: {messages}")

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(f"Generated response for index {i}: {response}")

                result = {
                    "index": i,
                    "video_name": video_name,
                    "messages": messages,
                    "response": response
                }

                f.write(json.dumps(result) + '\n')
                f.flush()

                processed_videos.add(video_name)

            except Exception as e:
                print(f"Error processing index {i}: {e}")