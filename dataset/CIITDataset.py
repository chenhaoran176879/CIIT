import copy
import json
import logging
import os
import pathlib
import warnings
import re
from collections import defaultdict

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image

def is_img(data):
    if isinstance(data,list):
        return all(isinstance(elem,Image.Image) for elem in data)
    return isinstance(data,Image.Image)

def is_text(data):
    if isinstance(data,list):
        return all(isinstance(elem,str) for elem in data)
    return isinstance(data,str)

baidu_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/baidujingyan_tuwen_data_2023-12-06.jsonl"
toutiao_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/tuwen_2023-11-02.jsonl"
   
'''
    {"title": "画图3D如何制作3D涂鸦字体", 
    "news_date": "2020-12-03 11:36", 
    "platform": "toutiao_tuwen", 
    "content": "我们在用画图3D制作字体的时候，我们可以使用3D涂鸦效果来制作立体字体，下面几步教会你！\n
    工具/原料\n电脑系统：windows10 家庭版\n系统类别：64位操作系统\n方法/步骤\n1.打开带有win10系统的电脑，打开程序，点击画图3D；
    如图\n<image>e2aefe781431dfb657ff19d012cf02532e636329</image>\n
    2.点击新建文件，点击裁剪，选择一个你想要的尺寸；如图\n<image>54a89daee8d7592a827caece9f31dfb6336c6729</image>\n
    3.点击3D形状，点击3D涂鸦，选择第一个画笔；如图\n<image>82eff6d7592ae3ef82eb5a8754b6326c56666429</image>\n
    4.更改涂鸦画笔的粗细，并选择一种你喜欢的颜色；如图\n<image>586bfdefe0781431bb257cdadc6699cf03536229</image>\n
    5.直接在画布上写上数字，可以调整锚点来更改角度；如图\n<image>e1390a31dfb6326c31f5d77989532f6322856029</image>\n
    6.再次把画笔的粗细调大，更改颜色，再次绘制；如图\n<image>332d496699cf025346f06d336b36e29146e85f29</image>\n
    注意事项\n如果对你有帮助请点赞关注", 
    "news_class": "",
      "news_url": "https://jingyan.baidu.com/article/200957619b26308a0621b41f.html", 
      "AIGC_info": [{"video_urls": [], "img_urls":
        ["https://exp-picture.cdn.bcebos.com/e2aefe781431dfb657ff19d012cf02532e636329.jpg",
        "https://exp-picture.cdn.bcebos.com/54a89daee8d7592a827caece9f31dfb6336c6729.jpg", 
        "https://exp-picture.cdn.bcebos.com/82eff6d7592ae3ef82eb5a8754b6326c56666429.jpg", 
      "https://exp-picture.cdn.bcebos.com/586bfdefe0781431bb257cdadc6699cf03536229.jpg",
        "https://exp-picture.cdn.bcebos.com/e1390a31dfb6326c31f5d77989532f6322856029.jpg", 
        "https://exp-picture.cdn.bcebos.com/332d496699cf025346f06d336b36e29146e85f29.jpg"], 
        "hash_codes": ["e2aefe781431dfb657ff19d012cf02532e636329", "54a89daee8d7592a827caece9f31dfb6336c6729",
          "82eff6d7592ae3ef82eb5a8754b6326c56666429", "586bfdefe0781431bb257cdadc6699cf03536229", "e1390a31dfb6326c31f5d77989532f6322856029", 
          "332d496699cf025346f06d336b36e29146e85f29"],
            "img_save_dir": "/home/yidongyi/ImageDataSets/dingchenglin_imgs/baidujingyan_tuwen/0"}],
            "create_time": "2023-11-22 10:14:14"}

''' 

def get_interleave_form(text):
    # 定义正则表达式模式来匹配<image>标记及链接部分
    pattern = r'<image>[^<]+</image>'

    # 使用findall()方法找到所有匹配的<image>标记及链接部分
    matches = re.findall(pattern, text)

    # 对于每个匹配项，提取<image>标记并用其前后的文本进行分割
    result = []
    last_end = 0
    for match in matches:
        start_index = text.find(match)
        if start_index > last_end:
            result.append(text[last_end:start_index])
        result.append(match)
        last_end = start_index + len(match)

    # 添加最后一个匹配项之后的文本
    if last_end < len(text):
        result.append(text[last_end:])

    return result


class CIITDataset(Dataset):
    def __init__(self, file_path_list=None,transforms=None):
        self.data = []
        if not file_path_list:
            file_path_list = [baidu_file,toutiao_file]
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))

        self.img_path = ''
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # returns texts in list and img in PIL.Image
        meta_info = self.data[idx]
        text = meta_info['content']
        if meta_info.get('title',None):
            text = meta_info['title'] + '\n' + text
        
        text_interleaved = get_interleave_form(text)
        image_save_dir = os.path.join(self.img_path, meta_info['AIGC_info'][0]['img_save_dir'])

        multimodal_context = []
        for modality in text_interleaved:
            if modality.startswith('<image>') and modality.endswith('</image>'):
                img_id = modality[len("<image>"):-len("</image>")]
                img_path = os.path.join(image_save_dir,img_id+'.jpg')
                try:
                    image = Image.open(img_path).convert("RGB")
                    
                except Exception as e:
                    print(f'Value Error: {img_path} does not exist or cannot open.')
                    continue
                
                if self.transforms:
                    image = self.transforms(image)
                multimodal_context.append(image)
            else:
                multimodal_context.append(modality)
        return  {'multimodal_context':multimodal_context, 'meta_info':meta_info}


    def get_meta_info(self,idx):
        return self.data[idx]


class CIITDataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'vae_image' in instances[0]:
            pil_images = [instance['vae_image'] for instance in instances]
            batch['vae_images'] = pil_images
        return batch


# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         list_data_dict = json.load(open(data_path, "r"))

#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split())
#                                for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split())
#                           for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(
#             sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(
#                 image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(
#                             pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(
#                             pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255)
#                                       for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')[
#                     'pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')[
#                     'pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(
#                 3, crop_size['height'], crop_size['width'])
#         return data_dict


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
#                                           data_path=data_args.data_path,
#                                           data_args=data_args)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)

