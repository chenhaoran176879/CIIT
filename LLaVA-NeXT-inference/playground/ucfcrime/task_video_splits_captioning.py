
# from .demo_modelpart import InferenceDemo
import gradio as gr
import os
# import time
import cv2
from PIL import Image
import json
# import copy
import torch
# import random
import numpy as np

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images


import requests
import os
from io import BytesIO
from transformers import TextStreamer
from transformers.generation.streamers import TextIteratorStreamer

#from llava.playground.ucfcrime.video_sampler import *

def is_valid_video_filename(name):
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    ext = name.split('.')[-1].lower()
    
    if ext in video_extensions:
        return True
    else:
        return False


def sample_frames_from_video_with_splits(video_file, num_frames,num_splits,start_frame=None,end_frame=None):
    # extract m splits , each with n frames from raw video
    # total_frames = num frames of raw video
    # total_num_frames = m * n
    video = cv2.VideoCapture(video_file)
    if start_frame and end_frame:
        total_frames = end_frame-start_frame+1
        video.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    else:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #duration = total_frames / fps
    total_num_frames = num_splits * num_frames

    
    if total_num_frames > total_frames:
        interval = 1
        num_frames = total_frames//num_splits
    else: 
        interval = total_frames // total_num_frames

    

    print(f'{total_frames=}\n{total_num_frames=}\n{interval=}\n{num_frames=}\n')


    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)        
    video.release()

    if len(frames) % num_frames != 0:
        frames = frames[:len(frames) // num_frames * num_frames]

    # 将frames划分为多个列表，每个列表包含num_frames个元素
    if start_frame:
        video_splits = [frames]
    else:
        video_splits = [frames[i * num_frames:(i + 1) * num_frames] for i in range(len(frames) // num_frames)]
    

    return video_splits



def get_prompt(args,image_token: str):
    model_name = get_model_name_from_path(args.model_path)
    conv_mode = None
    if 'qwen' in model_name.lower():
        conv_mode = "qwen_1_5"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))

    conversation = conv_templates[conv_mode].copy()

    user_input_text = ''
    user_multimodal_input = image_token + '\n' + user_input_text
    conversation.append_message(conversation.roles[0],user_multimodal_input)
    conversation.append_message(conversation.roles[1],None)
    prompt = conversation.get_prompt()

    return prompt

def video_captioning(args,model,image_processor,tokenizer,video_path,start_frame=None,end_frame=None):
    if is_valid_video_filename(video_path):
        frame_splits_list = sample_frames_from_video_with_splits(video_path,args.num_frames,args.num_splits,start_frame,end_frame)
    splits_caption_gather = []
    for i,frames in enumerate(frame_splits_list):
        #process_images()
        image_tensor = [image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0].half().to(model.device) for f in frames]
        image_tensor = torch.stack(image_tensor)
        image_token = DEFAULT_IMAGE_TOKEN*len(frames)

        assert len(frames)>0, "empty frames"

        model_name = get_model_name_from_path(args.model_path)
        conv_mode = None
        if 'qwen' in model_name.lower():
            conv_mode = "qwen_1_5"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))

        conversation = conv_templates[conv_mode].copy()

        user_input_text = 'We detected a crime act of abuse between people or people and animals in this video. Try to find it and describe the action. '
        user_multimodal_input = image_token + '\n' + user_input_text
        conversation.append_message(conversation.roles[0],user_multimodal_input)
        conversation.append_message(conversation.roles[1],None)
        prompt = conversation.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        image_sizes = [frame.size for frame in frames]
        stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)
        with torch.inference_mode():   
            cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs[0])
        splits_caption_gather.append(text_outputs[0])


    return splits_caption_gather


def main(args):

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name, args.load_8bit, args.load_4bit)
    video_file_path = os.path.dirname(args.train_data_path)

    caption_gather_cache = []
    output_file = '/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/video_caption_train.jsonl'
    output_f = open(output_file,'a',encoding='utf-8')
    with open(args.train_data_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            video_path = os.path.join(video_file_path,line)
            captions = video_captioning(args,model,image_processor,tokenizer,video_path)
            data_dict = {'video_name':line, 'captions':captions}
            print(data_dict)
            json.dump(data_dict,output_f)
            output_f.write('\n')

    output_f.close()
        
    return



def annotate_test_split(args):
    test_data_path = '/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'

    assert args.num_splits == 1, "test video should not be splited"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name, args.load_8bit, args.load_4bit)
    video_file_path = os.path.dirname(test_data_path)

    caption_gather = []
    output_file = '/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/video_caption_test.jsonl'
    output_f = open(output_file,'a',encoding='utf-8')
    with open(test_data_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            fields = line.split()
            entry = {
            "filename": fields[0],
            "category": fields[1],
            "start_frame": int(fields[2]),
            "end_frame": int(fields[3]),
            "start_frame_1": int(fields[4]),
            "end_frame_1": int(fields[5])
        }
            if entry['category'] == 'Normal':
                continue

            
            video_path = os.path.join(video_file_path,entry['category'],entry['filename'])
            print(video_path)
            try:
                captions = video_captioning(args,model,image_processor,tokenizer,video_path,entry['start_frame'],entry['end_frame'])
            except Exception as e:
                print("Exception:",e)
                continue
            data_dict = {'video_name':os.path.join(entry['category'],entry['filename']), 'start_frame':entry['start_frame'],'end_frame':entry['end_frame'],'captions':captions}
            print(data_dict)
            json.dump(data_dict,output_f)
            output_f.write('\n')
            
    output_f.close()

    return
def test_main(args):
    image_token = DEFAULT_IMAGE_TOKEN*10
    prompt = get_prompt(args,image_token)
    print(prompt)
    return 



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6125", type=str)
    argparser.add_argument("--model_path", default="/mnt/lustre/chenhaoran/model_zoo/llava-next-interleave-qwen-7B/llava-next-interleave-qwen-7b/", type=str)
    # argparser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument('--mode',type=str,default='test')
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--num_frames", type=int, default=32)
    argparser.add_argument("--num_splits", type=int, default=6)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument('--train_data_path',type=str,default='Anomaly_Train.txt')

    args = argparser.parse_args()
    model_name = get_model_name_from_path(args.model_path)
    args.model_name = model_name

    if args.mode == 'train':
        main(args)
    elif args.mode == 'test':
        annotate_test_split(args)
    


