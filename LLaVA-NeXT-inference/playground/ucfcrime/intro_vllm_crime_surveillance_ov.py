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
question = f"{DEFAULT_IMAGE_TOKEN}\nWe detected a crime act of abuse between people or people and animals in this video. Try to find it and describe the action."

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


# Describe what's happening in this video.
# The video takes place in a spacious, well-lit room with high ceilings and large windows. The room features a large painting on the wall behind a table, which is positioned on a gray carpeted area. A person wearing a light blue jacket and dark pants stands at the table, seemingly engaged in an activity that involves handling some papers or documents. Another individual, dressed in a black jacket and dark pants, enters the scene from the right side and approaches the table. This second person appears to be assisting or interacting with the first person, possibly discussing or organizing the items on the table. The background includes two closed doors and a staircase leading to an upper level.

# As the scene continues, the person in the light blue jacket remains at the table, while the individual in the black jacket moves around the table, occasionally bending down as if picking up or placing items. The interaction between the two individuals suggests they are working together on the task at hand. The background remains consistent with the closed doors and staircase.

# In the final part of the video, the person in the light blue jacket is now lying on the floor near the table, appearing to be in distress or discomfort. The individual in the black jacket is seen bending over the person on the floor, possibly checking on them or offering assistance. The background remains unchanged, with the closed doors and staircase still visible. The overall atmosphere of the video suggests a collaborative effort initially, followed by a moment of concern or assistance when one of the individuals falls to the ground.



# We detected a crime act of abuse between people or people and animals in this video. Try to find it and describe the action.
#视频开始时，一个人站在一个大而空的房间里的桌子后面，房间里有高高的天花板和拱形的门。此人身穿浅蓝色夹克和深色裤子。另一个人穿着黑色衣服从右侧进入画面。第二个人走近桌子，与桌子上的物品互动，导致纸张散落在地板上。第一个人在整个互动过程中保持静止。随着黑衣人继续与地板上散落的纸张互动，场景逐渐展开，而第一个人则留在桌子后面。故事继续进行，黑衣人仍在与地板上散落的纸张互动，而第一个人则留在桌子后面。视频的最后一部分显示，黑衣人跪在桌子旁边，似乎在整理或捡起散落的文件。第一个人站在桌子后面不动，观察情况。