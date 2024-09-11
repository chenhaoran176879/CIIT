model_name = None

questions = [
    'Describe this video.',
    'Is there any act of abuse?',
    'Did you see a cat and a lady?',
    'What did the lady do to the cat?',
    'Is this an act of abuse?'
]

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

video_dataset = '/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/'
video_meta = {"video_name": "Abuse001_x264.mp4", "start_time": 4, "end_time": 18, "description": "\u89c6\u9891\u4e2d\u53d1\u751f\u4e86\u66b4\u529b\u62a2\u52ab\u548c\u8650\u5f85\u6bb4\u6253\u7684\u72af\u7f6a\u884c\u4e3a\u3002\u5728\u4e00\u4e2a\u5927\u7406\u77f3\u5927\u5385\u4e2d\uff0c\u4e00\u4f4d\u94f6\u8272\u77ed\u53d1\u3001\u8eab\u7740\u767d\u8272\u8863\u670d\u3001\u9ed1\u8272\u88e4\u5b50\u7684\u5973\u58eb\u6b63\u5728\u5927\u5385\u4e2d\u592e\u7684\u6728\u684c\u524d\u9605\u62a5\uff0c\u4e24\u540d\u72af\u7f6a\u5206\u5b50\u4ece\u5de6\u4e0b\u89d2\u4fb5\u5165\uff0c\u7b2c\u4e00\u540d\u8eab\u7a7f\u767d\u8272T\u6064\u7684\u9ed1\u4eba\u62a2\u8d70\u4e86\u5973\u58eb\u7684\u7ea2\u8272\u8170\u5305\uff0c\u7b2c\u4e8c\u540d\u8eab\u7a7f\u9ed1\u8272T\u6064\u7684\u9ed1\u4eba\u7528\u62f3\u5934\u51fb\u6253\u5973\u58eb\u7684\u5934\u90e8\uff0c\u5bfc\u81f4\u5973\u58eb\u5012\u5730\u5e76\u75db\u82e6\u5730\u6342\u8138\uff0c\u968f\u540e\u5750\u8d77\u3002", "quality_score": 8, "confidence_score": 10, "timestamp": "2024-08-29 09:32:28", "is_repeated": "\u662f", "first_end_time": 23}


video_path = os.path.join(video_dataset, video_meta['video_name'][:-12], video_meta['video_name'])
video_frames = load_video(video_path, 32, start_time=video_meta['start_time'], end_time=video_meta['end_time'])




pretrained = "/home/share/chenhaoran/model_zoo/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd/"
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

image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]
modalities = ["video"] * len(video_frames)

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
print(text_outputs[0])
