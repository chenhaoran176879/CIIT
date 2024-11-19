from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
from transformers import AutoConfig
import os
import json



warnings.filterwarnings("ignore")


from decord import VideoReader, cpu
import numpy as np

def load_video(video_path, max_frames_num, start_time=0, end_time=None):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))

    fps = vr.get_avg_fps() 
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else len(vr)
    
    start_frame = max(0, start_frame)
    end_frame = min(len(vr), end_frame)

    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames




pretrained = "/home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

config = AutoConfig.from_pretrained(pretrained)

config.mm_vision_tower = "/home/share/models/siglip-so400m-patch14-384/"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa",customized_config=config
)
model = model.to(device)
model.eval()


conv_template = "qwen_1_5"

conv = copy.deepcopy(conv_templates[conv_template])


def chat_with_llavaov_model(question, video_frames, image_tensors, video_path=None, nframes=32, max_tokens=128):
    global conv


    image_sizes = [frame.size for frame in video_frames]
    modalities = ["video"] * len(video_frames)
    if len(conv.messages) == 0:
        question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    #conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt_question = conv.get_prompt()

    print(prompt_question)
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=modalities,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)


    if conv.messages[-1][0] == conv.roles[1]:
        conv.messages[-1][1] = text_outputs[0]
        
    return text_outputs[0]



def ask_questions(questions, video_path, nframes=32, max_tokens=128,model_name=pretrained.split('/')[-1]):
    global messages
    responses = []
    video_dir, video_filename = os.path.split(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_dir = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/"
    output_file = os.path.join(output_dir, f"{model_name}_{video_name}_{nframes}frames.json")
    # Check if the output file exists; if not, create an empty JSON array in it
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # # Load existing history if any
    # with open(output_file, 'r') as f:
    #     dialogue_history = json.load(f)
    
    # Iterate through questions, ask the model, and store responses
    
    dialogue_history = []
    print(f"Processing {video_name}")
    video_frames = load_video(video_path, nframes)#,start_time=video_meta['start_time'], end_time=video_meta['end_time'])
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    for question in questions:
        response = chat_with_llavaov_model(question, video_frames=video_frames,image_tensors=image_tensors,video_path=video_path, nframes=nframes, max_tokens=max_tokens)
        
        # Print out for verification
        print("User:", question)
        print("Assistant:", response)
        
        # Append response to list
        responses.append(response)
        
        # Save each Q&A pair to the dialogue history
        dialogue_entry = {
            "question": question,
            "response": response
        }
        dialogue_history.append(dialogue_entry)
        
        # Write updated history back to the file
    with open(output_file, 'w') as f:
        json.dump(dialogue_history, f, ensure_ascii=False, indent=4)
    
    messages = []
    
    return responses

# Example usage


def generate_video_paths(videos, base_dir="/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train"):
    video_paths_dict = {}
    
    for video in videos:
        # Extract category from video name
        category = video.split('0')[0]  # Extract category like 'Assault' from 'Assault011'
        
        # Original video path
        video_path = os.path.join(base_dir, category, f"{video}_x264.mp4")
        
        # Generate trimmed video path
        trimmed_video_path = os.path.join(base_dir, "crime_videos_cut", f"trimmed_{video}_x264.mp4")
        
        # Store both paths in the dictionary with video name as the key
        video_paths_dict[video] = {
            'original': video_path,
            'trimmed': trimmed_video_path
        }
    
    return video_paths_dict

videos_to_test = ['Assault011','Arrest001','Fighting003','Explosion021']
video_paths = generate_video_paths(videos_to_test)

with open("/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/stepbystep_questions.json",'r') as f:
    questions = json.load(f)


for video in videos_to_test:
    questions_per_video = questions[video]
    trimmed = video_paths[video]['trimmed']
    original = video_paths[video]['original']
    conv = copy.deepcopy(conv_templates[conv_template])
    responses = ask_questions(questions[video], trimmed, nframes=32)
    torch.cuda.empty_cache()

    conv = copy.deepcopy(conv_templates[conv_template])
    responses = ask_questions(questions[video], original, nframes=32)
    torch.cuda.empty_cache()
