

import torch
from transformers import AutoTokenizer, AutoModel
path = "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-40B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
generation_config = dict(max_new_tokens=1024, do_sample=False)

from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import torchvision.transforms as T
import os
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.chat(inputs.input_ids, max_length=200, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response







def ask_questions(questions, video_path, nframes=32, max_tokens=128,model_name=path.split('/')[-1]):
    responses = []
    video_dir, video_filename = os.path.split(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_dir = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/"
    output_file = os.path.join(output_dir, f"{model_name}_{video_name}_{nframes}frames.json")
    # Check if the output file exists; if not, create an empty JSON array in it
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Load existing history if any
    with open(output_file, 'r') as f:
        dialogue_history = json.load(f)
    
    pixel_values, num_patches_list = load_video(video_path, num_segments=nframes, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    history = None
    # Iterate through questions, ask the model, and store responses
    print(f"Processing {video_name}")
    for question in questions:
        if history ==None:
            question = video_prefix+question

        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=history, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        
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

videos_to_test = ['Fighting003','Assault011','Arrest001','Explosion021']
video_paths = generate_video_paths(videos_to_test)

with open("/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/crime_videos_cut/stepbystep_questions.json",'r') as f:
    questions = json.load(f)


for video in videos_to_test:
    questions_per_video = questions[video]
    trimmed = video_paths[video]['trimmed']
    original = video_paths[video]['original']
    responses = ask_questions(questions[video], original, nframes=32)
    torch.cuda.empty_cache()
    responses = ask_questions(questions[video], trimmed, nframes=32)
    torch.cuda.empty_cache()
