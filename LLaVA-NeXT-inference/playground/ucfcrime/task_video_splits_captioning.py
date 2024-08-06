
# from .demo_modelpart import InferenceDemo
import gradio as gr
import os
# import time
import cv2
from PIL import Image

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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import requests

from io import BytesIO
from transformers import TextStreamer

#from llava.playground.ucfcrime.video_sampler import *

def is_valid_video_filename(name):
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    ext = name.split('.')[-1].lower()
    
    if ext in video_extensions:
        return True
    else:
        return False


def sample_frames_from_video_with_splits(video_file, num_frames,num_splits):
    # extract m splits , each with n frames from raw video
    # total_frames = num frames of raw video
    # total_num_frames = m * n
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #duration = total_frames / fps
    total_num_frames = num_splits * num_frames
    interval = total_frames // total_num_frames
    
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
    video_splits = [frames[i * num_frames:(i + 1) * num_frames] for i in range(len(frames) // num_frames)]
    

    return video_splits



def video_captioning(args,model,video_path):
    if is_valid_video_filename(video_path):
        frame_splits_list = sample_frames_from_video_with_splits(video_path,args.num_frames,args.num_splits)


    return





if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6125", type=str)
    argparser.add_argument("--model_path", default="", type=str)
    # argparser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--num_frames", type=int, default=32)
    argparser.add_argument("--num_splits", type=int, default=32)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")


    args = argparser.parse_args()
    model_path = args.model_path
    filt_invalid="cut"
    
    model_name = get_model_name_from_path(args.model_path)
    print("get_model_name:",model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    print("loaded from pretrain")
   

