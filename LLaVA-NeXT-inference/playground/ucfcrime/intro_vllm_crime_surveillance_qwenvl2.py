from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os

# Load the model and processor
MODEL_PATH = "/home/share/chenhaoran/model_zoo/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Initialize conversation history
messages = []

def chat_with_qwen2vl_model(question, video_path=None, nframes=32, max_tokens=128):
    global messages  # Access the global messages list for conversation history

    # Append new user question with video context to conversation history
    if not messages:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": nframes,
                },
                {"type": "text", "text": question},
            ],
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        })

    # Prepare text and inputs for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Append model response to conversation history
    messages.append({"role": "assistant", "content": output_text})

    return output_text

def ask_questions(questions, video_path, nframes=32, max_tokens=128,model_name=MODEL_PATH.split('/')[-1]):
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
    
    # Load existing history if any
    with open(output_file, 'r') as f:
        dialogue_history = json.load(f)
    
    # Iterate through questions, ask the model, and store responses
    print(f"Processing {video_name}")
    for question in questions:
        response = chat_with_qwen2vl_model(question, video_path=video_path, nframes=nframes, max_tokens=max_tokens)
        
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
    responses = ask_questions(questions[video], trimmed, nframes=64)
    torch.cuda.empty_cache()
    responses = ask_questions(questions[video], original, nframes=64)
    torch.cuda.empty_cache()
